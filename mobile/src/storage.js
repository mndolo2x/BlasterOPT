/**
 * Offline-first Local Storage Module for BlastOpt Mobile.
 *
 * Implements a thread-safe local JSON storage store for caching blast designs,
 * local field logs, and pending sync queues when operating in remote open-pit mine locations.
 */

class LocalStorageStore {
  constructor() {
    this.store = {
      designs: [
        { id: 'DES_JWA_001', name: 'Jwaneng Cut 8 Bench 15S', holes: 24, pf: 0.65, status: 'APPROVED' },
        { id: 'DES_ORA_002', name: 'Orapa Pit 1 Bench 12N', holes: 36, pf: 0.72, status: 'APPROVED' },
      ],
      fieldLogs: [],
      syncQueue: [],
    };
  }

  getItem(key) {
    return this.store[key] || null;
  }

  setItem(key, value) {
    this.store[key] = value;
  }

  clear() {
    this.store = { designs: [], fieldLogs: [], syncQueue: [] };
  }
}

export const localStorage = new LocalStorageStore();
