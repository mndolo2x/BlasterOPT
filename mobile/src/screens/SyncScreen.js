import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView } from 'react-native';
import { getSyncQueue, replaySyncQueue } from '../sync';

export default function SyncScreen({ role }) {
  const [queue, setQueue] = useState([]);
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    setQueue(getSyncQueue());
  }, []);

  const handleTriggerSync = () => {
    if (!isOnline) {
      alert('Cannot sync: Mobile device is currently offline.');
      return;
    }
    const result = replaySyncQueue();
    setQueue(getSyncQueue());
    alert(`Sync Complete:\nProcessed ${result.replayedCount} queued field logs with zero conflicts.`);
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Offline Sync Queue & Status</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Network Connection Status</Text>
        <TouchableOpacity
          style={[styles.toggleButton, isOnline ? styles.onlineButton : styles.offlineButton]}
          onPress={() => setIsOnline(!isOnline)}
        >
          <Text style={styles.toggleText}>
            Status: {isOnline ? 'ONLINE (4G/LTE)' : 'OFFLINE (Remote Pit)'}
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Pending Sync Queue ({queue.length} items)</Text>
        {queue.length === 0 ? (
          <Text style={styles.cardText}>Queue is empty. All local field logs are synchronized with cloud.</Text>
        ) : (
          queue.map((item, idx) => (
            <View key={idx} style={styles.queueItem}>
              <Text style={styles.queueText}>#{idx + 1} Hole: {item.payload.holeId} (Depth: {item.payload.depth}m, Charge: {item.payload.chargeWeight}kg)</Text>
            </View>
          ))
        )}

        <TouchableOpacity style={styles.syncButton} onPress={handleTriggerSync}>
          <Text style={styles.syncButtonText}>Replay & Reconcile Sync Queue</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 12, color: '#333' },
  card: { backgroundColor: '#fff', borderRadius: 8, padding: 16, marginBottom: 12 },
  cardTitle: { fontSize: 16, fontWeight: 'bold', marginBottom: 8, color: '#1976D2' },
  cardText: { fontSize: 13, color: '#666' },
  toggleButton: { padding: 12, borderRadius: 6, alignItems: 'center' },
  onlineButton: { backgroundColor: '#e8f5e9' },
  offlineButton: { backgroundColor: '#ffebee' },
  toggleText: { fontWeight: 'bold', color: '#333' },
  queueItem: { paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#eee' },
  queueText: { fontSize: 13, color: '#444' },
  syncButton: { backgroundColor: '#1976D2', borderRadius: 6, padding: 14, alignItems: 'center', marginTop: 14 },
  syncButtonText: { color: '#fff', fontWeight: 'bold' },
});
