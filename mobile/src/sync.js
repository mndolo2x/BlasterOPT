/**
 * Offline Sync Queue, Write-Ahead Log (WAL) & Validation Module for BlastOpt Mobile.
 *
 * Implements field log validation rules, persistent Write-Ahead Log (WAL) queuing,
 * exponential backoff retry calculations, and conflict resolution strategies.
 */

import { localStorage } from './storage.js';

/**
 * Validates field log input parameters against domain safety bounds.
 *
 * @param {Object} log - Field log payload {holeId, depth, chargeWeight, stemming}.
 * @returns {Object} { isValid: boolean, errors: Array<string> }
 */
export function validateFieldLog(log) {
  const errors = [];

  if (!log || typeof log !== 'object') {
    return { isValid: false, errors: ['Invalid field log payload.'] };
  }

  if (!log.holeId || typeof log.holeId !== 'string' || log.holeId.trim() === '') {
    errors.push('Hole ID is required.');
  }

  if (typeof log.depth !== 'number' || isNaN(log.depth) || log.depth < 1.0 || log.depth > 50.0) {
    errors.push('Measured Depth must be between 1.0 m and 50.0 m.');
  }

  if (typeof log.chargeWeight !== 'number' || isNaN(log.chargeWeight) || log.chargeWeight < 1.0 || log.chargeWeight > 2000.0) {
    errors.push('Charge Weight must be between 1.0 kg and 2000.0 kg.');
  }

  if (typeof log.stemming !== 'number' || isNaN(log.stemming) || log.stemming < 0.5 || log.stemming > 15.0) {
    errors.push('Stemming Length must be between 0.5 m and 15.0 m.');
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
}

/**
 * Calculates exponential backoff delay in milliseconds.
 *
 * @param {number} attempt - Zero-based retry attempt index.
 * @param {number} baseDelayMs - Base delay in milliseconds (default: 1000ms).
 * @param {number} backoffFactor - Backoff factor (default: 2.0).
 * @returns {number} Backoff delay in milliseconds.
 */
export function calculateBackoffDelay(attempt, baseDelayMs = 1000, backoffFactor = 2.0) {
  return baseDelayMs * Math.pow(backoffFactor, attempt);
}

/**
 * Resolves concurrent modification conflicts between local edits and remote records.
 *
 * @param {Object} local - Local field edit record.
 * @param {Object} remote - Remote server-side record.
 * @param {string} strategy - Conflict strategy ('last_write_wins', 'remote_wins', 'local_wins').
 * @returns {Object} Resolved record.
 */
export function resolveConflicts(local, remote, strategy = 'last_write_wins') {
  if (!local && !remote) return {};
  if (!local) return { ...remote };
  if (!remote) return { ...local };

  if (strategy === 'remote_wins') return { ...remote };
  if (strategy === 'local_wins') return { ...local };

  const localTs = new Date(local.timestamp || local.updatedAt || '1970-01-01').getTime();
  const remoteTs = new Date(remote.timestamp || remote.updatedAt || '1970-01-01').getTime();

  if (localTs >= remoteTs) {
    return { ...local, conflictResolvedBy: 'last_write_wins_local' };
  } else {
    return { ...remote, conflictResolvedBy: 'last_write_wins_remote' };
  }
}

/**
 * Enqueues an offline action into the persistent Write-Ahead Log (WAL) sync queue.
 *
 * @param {Object} action - Action object { type: string, payload: Object }.
 */
export function enqueueSyncAction(action) {
  const queue = localStorage.getItem('syncQueue') || [];
  queue.push({
    id: `WAL_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
    timestamp: new Date().toISOString(),
    attempts: 0,
    ...action,
  });
  localStorage.setItem('syncQueue', queue);
}

/**
 * Retrieves current pending Write-Ahead Log (WAL) queue items.
 *
 * @returns {Array<Object>} List of queued sync actions.
 */
export function getSyncQueue() {
  return localStorage.getItem('syncQueue') || [];
}

/**
 * Replays and reconciles queued sync actions upon network reconnection.
 *
 * @returns {Object} { replayedCount: number, status: string, lastSyncTimestamp: string }
 */
export function replaySyncQueue() {
  const queue = getSyncQueue();
  const replayedCount = queue.length;

  // Process queued logs into local fieldLogs store
  const logs = localStorage.getItem('fieldLogs') || [];
  queue.forEach((item) => {
    if (item.type === 'FIELD_LOG') {
      logs.push(item.payload);
    }
  });

  localStorage.setItem('fieldLogs', logs);
  localStorage.setItem('syncQueue', []); // Clear WAL queue after replay
  const lastSyncTimestamp = new Date().toISOString();
  localStorage.setItem('lastSyncTimestamp', lastSyncTimestamp);

  return {
    replayedCount,
    status: 'SYNC_SUCCESS',
    lastSyncTimestamp,
  };
}
