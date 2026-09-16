/**
 * Unit tests for offline sync queue replay logic.
 */

import { enqueueSyncAction, getSyncQueue, replaySyncQueue } from '../src/sync.js';
import { localStorage } from '../src/storage.js';

describe('Offline Sync Queue Manager', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  test('enqueueSyncAction appends items to sync queue', () => {
    enqueueSyncAction({ type: 'FIELD_LOG', payload: { holeId: 'BH-201', depth: 15.0 } });
    const queue = getSyncQueue();

    expect(queue.length).toBe(1);
    expect(queue[0].type).toBe('FIELD_LOG');
    expect(queue[0].payload.holeId).toBe('BH-201');
  });

  test('replaySyncQueue processes queued logs and clears queue upon sync', () => {
    enqueueSyncAction({ type: 'FIELD_LOG', payload: { holeId: 'BH-201', depth: 15.0 } });
    enqueueSyncAction({ type: 'FIELD_LOG', payload: { holeId: 'BH-202', depth: 15.2 } });

    expect(getSyncQueue().length).toBe(2);

    const result = replaySyncQueue();

    expect(result.status).toBe('SYNC_SUCCESS');
    expect(result.replayedCount).toBe(2);
    expect(getSyncQueue().length).toBe(0);

    const logs = localStorage.getItem('fieldLogs');
    expect(logs.length).toBe(2);
    expect(logs[0].holeId).toBe('BH-201');
  });
});
