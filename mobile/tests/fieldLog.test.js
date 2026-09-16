/**
 * Unit tests for field log validation rules.
 */

import { validateFieldLog } from '../src/sync.js';

describe('Field Log Input Validation', () => {
  test('validates compliant field log payload', () => {
    const validLog = {
      holeId: 'BH-105',
      depth: 15.0,
      chargeWeight: 320.0,
      stemming: 5.0,
    };

    const res = validateFieldLog(validLog);
    expect(res.isValid).toBe(true);
    expect(res.errors.length).toBe(0);
  });

  test('flags invalid holeId, out-of-range depth, charge weight, and stemming', () => {
    const invalidLog = {
      holeId: '',
      depth: 99.0, // > 50.0m
      chargeWeight: -10.0, // < 1.0kg
      stemming: 0.1, // < 0.5m
    };

    const res = validateFieldLog(invalidLog);
    expect(res.isValid).toBe(false);
    expect(res.errors.length).toBe(4);
    expect(res.errors[0]).toContain('Hole ID is required');
    expect(res.errors[1]).toContain('Measured Depth');
    expect(res.errors[2]).toContain('Charge Weight');
    expect(res.errors[3]).toContain('Stemming Length');
  });
});
