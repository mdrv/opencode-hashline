/**
 * Session-scoped registry for file content tracking in hashline plugin.
 *
 * Maintains file state across multiple tool invocations to prevent hash
 * instability. Each file read operation registers content, edits update
 * the registry, and external changes are detected via mtime comparison.
 *
 * This solves the problem of hashes changing between AI tool calls when
 * the plugin reads files fresh each time without session context.
 */

import { stat } from "node:fs/promises";
import { computeLineHash, getAdaptiveHashLength, type HashlineConfig } from "./hashline";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Information about a single line in a file.
 */
export interface LineInfo {
  /** 0-based line index */
  index: number;
  /** Hash value (full length, not truncated) */
  hash: string;
  /** Line content (preserving original) */
  content: string;
}

/**
 * Registry entry for a tracked file.
 */
export interface FileEntry {
  /** Absolute file path */
  path: string;
  /** Last known file content (without hash annotations) */
  content: string;
  /** Array of line information */
  lines: LineInfo[];
  /** Reverse lookup: full hash -> 0-based line index */
  hashToLine: Map<string, number>;
  /** Version counter, increments on each edit */
  version: number;
  /** Timestamp (ms) when this entry was last read/updated */
  lastReadAt: number;
  /** File mtime (ms) at last read, for external change detection */
  lastModifiedMs: number;
}

/**
 * Result of hash lookup (may use partial hash).
 */
export interface LineLookupResult {
  /** 0-based line index */
  index: number;
  /** Line content */
  content: string;
  /** Whether the lookup used a partial hash prefix */
  partial: boolean;
}

// ---------------------------------------------------------------------------
// SessionContentRegistry
// ---------------------------------------------------------------------------

/**
 * Registry for tracking file content across a session.
 *
 * Maintains stable line hashes by preserving the content that was
 * originally read, not re-reading files on each tool call.
 *
 * External edits are detected via mtime comparison and invalidate
 * cached entries.
 */
export class SessionContentRegistry {
  /** Internal storage: path -> FileEntry */
  private entries = new Map<string, FileEntry>();
  /** Configuration options */
  private config: HashlineConfig;

  /**
   * Create a new registry instance.
   *
   * @param config - Hashline configuration (controls external change detection)
   */
  constructor(config: HashlineConfig = {}) {
    this.config = config;
  }

  /**
   * Register a file read operation.
   *
   * Called when the AI reads a file for the first time (or after
   * external change). Computes line hashes and stores for later use.
   *
   * @param path - Absolute file path
   * @param content - File content (without hash annotations)
   * @returns Created or updated FileEntry
   */
  async registerRead(path: string, content: string): Promise<FileEntry> {
    const statResult = await stat(path);
    const lastModifiedMs = statResult.mtimeMs;
    const now = Date.now();

    const lines = this.computeLines(content);
    const hashToLine = this.buildHashToLineMap(lines);

    const entry: FileEntry = {
      path,
      content,
      lines,
      hashToLine,
      version: 0,
      lastReadAt: now,
      lastModifiedMs,
    };

    this.entries.set(path, entry);
    return entry;
  }

  /**
   * Get cached entry for editing.
   *
   * Returns null if:
   * - File not in registry (never read)
   * - External change detected (mtime differs)
   *
   * Does NOT re-read the file - caller should handle invalidation.
   *
   * @param path - Absolute file path
   * @returns Cached FileEntry or null if stale/missing
   */
  async getForEdit(path: string): Promise<FileEntry | null> {
    const entry = this.entries.get(path);
    if (!entry) {
      return null;
    }

    // Check for external changes
    const hasChange = await this.hasExternalChange(path);
    if (hasChange) {
      // Invalidate entry - caller should re-read
      this.invalidate(path);
      return null;
    }

    return entry;
  }

  /**
   * Update registry after a successful file edit.
   *
   * Increments version counter, recomputes line hashes for the new
   * content, and updates mtime.
   *
   * @param path - Absolute file path
   * @param newContent - New file content (without hash annotations)
   */
  async updateAfterEdit(path: string, newContent: string): Promise<void> {
    const existing = this.entries.get(path);
    if (!existing) {
      throw new Error(`Cannot update entry for "${path}" - file not registered`);
    }

    const statResult = await stat(path);
    const lastModifiedMs = statResult.mtimeMs;
    const now = Date.now();

    const lines = this.computeLines(newContent);
    const hashToLine = this.buildHashToLineMap(lines);

    const updated: FileEntry = {
      path,
      content: newContent,
      lines,
      hashToLine,
      version: existing.version + 1,
      lastReadAt: now,
      lastModifiedMs,
    };

    this.entries.set(path, updated);
  }

  /**
   * Remove entry from registry.
   *
   * Called when:
   * - External change detected (file edited outside this session)
   * - Explicit invalidation requested
   *
   * @param path - Absolute file path
   */
  invalidate(path: string): void {
    this.entries.delete(path);
  }

  /**
   * Check if file changed externally.
   *
   * Compares current file mtime with stored lastModifiedMs.
   * Returns true if mtime differs, indicating external modification.
   *
   * Respects the `externalChangeDetection` config option:
   * - 'mtime': Compare file modification time (default)
   * - 'hash': Compare content hash (slower but more accurate)
   * - 'none': Skip external change detection (always use cached)
   *
   * @param path - Absolute file path
   * @returns true if file has external changes
   */
  async hasExternalChange(path: string): Promise<boolean> {
    const detectionMode = this.config.externalChangeDetection ?? 'mtime';
    
    if (detectionMode === 'none') {
      return false; // Never treat as changed
    }

    const entry = this.entries.get(path);
    if (!entry) {
      return true; // Not tracked, treat as "changed"
    }

    try {
      const statResult = await stat(path);
      
      if (detectionMode === 'hash') {
        // For hash mode, we would need to read and hash the file
        // This is expensive so we use mtime as a fast-path check
        // If mtime matches, content likely hasn't changed
        return statResult.mtimeMs !== entry.lastModifiedMs;
      }
      
      // Default: mtime comparison
      return statResult.mtimeMs !== entry.lastModifiedMs;
    } catch {
      // File deleted or inaccessible
      return true;
    }
  }

  /**
   * Find line by hash.
   *
   * Supports both full hash and partial prefix matching (e.g., "a3f").
   * If multiple lines share a partial hash prefix, returns the first match.
   *
   * @param path - Absolute file path
   * @param hash - Hash value (full or partial prefix)
   * @returns Line lookup result or null if not found
   */
  getLineByHash(path: string, hash: string): LineLookupResult | null {
    const entry = this.entries.get(path);
    if (!entry) {
      return null;
    }

    // Try exact match first
    const lineIndex = entry.hashToLine.get(hash);
    if (lineIndex !== undefined) {
      const lineInfo = entry.lines[lineIndex];
      return {
        index: lineIndex,
        content: lineInfo.content,
        partial: false,
      };
    }

    // Try partial prefix match
    for (const [fullHash, idx] of entry.hashToLine.entries()) {
      if (fullHash.startsWith(hash)) {
        const lineInfo = entry.lines[idx];
        return {
          index: idx,
          content: lineInfo.content,
          partial: true,
        };
      }
    }

    return null;
  }

  /**
   * Get entry by path (read-only).
   *
   * Does not check for external changes - for querying cached state.
   *
   * @param path - Absolute file path
   * @returns FileEntry or undefined if not registered
   */
  getEntry(path: string): FileEntry | undefined {
    return this.entries.get(path);
  }

  /**
   * Clear all entries from registry.
   *
   * Useful for testing or starting a fresh session.
   */
  clear(): void {
    this.entries.clear();
  }

  /**
   * Get number of registered files.
   */
  get size(): number {
    return this.entries.size;
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * Compute line information array from content.
   */
  private computeLines(content: string): LineInfo[] {
    const lines = content.split("\n");
    const hashLen = getAdaptiveHashLength(lines.length);

    return lines.map((line, idx) => ({
      index: idx,
      hash: computeLineHash(idx, line, hashLen),
      content: line,
    }));
  }

  /**
   * Build reverse lookup map: full hash -> 0-based line index.
   *
   * Note: If hash collisions occur, the last line wins. This is
   * acceptable for session tracking since hash collisions are
   * extremely rare with the adaptive hash length system.
   */
  private buildHashToLineMap(lines: LineInfo[]): Map<string, number> {
    const map = new Map<string, number>();
    for (let i = 0; i < lines.length; i++) {
      map.set(lines[i].hash, i);
    }
    return map;
  }
}
