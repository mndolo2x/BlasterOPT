/**
 * Offline Sync Queue & Validation Module for BlastOpt Mobile.
 *
 * Implements field log validation rules and offline action replay queue management.
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
 * Enqueues an offline action into the sync queue.
 *
 * @param {Object} action - Action object { type: string, payload: Object }.
 */
export function enqueueSyncAction(action) {
  const queue = localStorage.getItem('syncQueue') || [];
  queue.push({
    id: `SYNC_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
    timestamp: new Date().toISOString(),
    ...action,
  });
  localStorage.setItem('syncQueue', queue);
}

/**
 * Retrieves current pending sync queue items.
 *
 * @returns {Array<Object>} List of queued sync actions.
 */
export function getSyncQueue() {
  return localStorage.getItem('syncQueue') || [];
}

/**
 * Replays and reconciles queued sync actions upon network reconnection.
 *
 * @returns {Object} { replayedCount: number, status: string }
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
  localStorage.setItem('syncQueue', []); // Clear queue after replay

  return {
    replayedCount,
    status: 'SYNC_SUCCESS',
  };
}
