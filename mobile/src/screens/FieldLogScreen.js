import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { validateFieldLog, enqueueSyncAction } from '../sync';

export default function FieldLogScreen({ role }) {
  const [holeId, setHoleId] = useState('');
  const [depth, setDepth] = useState('');
  const [chargeWeight, setChargeWeight] = useState('');
  const [stemming, setStemming] = useState('');

  const handleSaveLog = () => {
    const payload = {
      holeId,
      depth: parseFloat(depth),
      chargeWeight: parseFloat(chargeWeight),
      stemming: parseFloat(stemming),
      timestamp: new Date().toISOString(),
      loggedByRole: role,
    };

    const validation = validateFieldLog(payload);
    if (!validation.isValid) {
      alert(`Validation Error:\n${validation.errors.join('\n')}`);
      return;
    }

    enqueueSyncAction({ type: 'FIELD_LOG', payload });
    alert('Success: Field log saved locally in offline sync queue.');
    setHoleId('');
    setDepth('');
    setChargeWeight('');
    setStemming('');
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Field Data Log Entry (Offline First)</Text>

      <View style={styles.card}>
        <Text style={styles.label}>Hole ID (e.g. BH-101)</Text>
        <TextInput style={styles.input} value={holeId} onChangeText={setHoleId} placeholder="BH-101" />

        <Text style={styles.label}>Measured Depth (m) [1.0 - 50.0m]</Text>
        <TextInput style={styles.input} value={depth} onChangeText={setDepth} keyboardType="numeric" placeholder="15.0" />

        <Text style={styles.label}>Charge Weight (kg) [1.0 - 2000.0kg]</Text>
        <TextInput style={styles.input} value={chargeWeight} onChangeText={setChargeWeight} keyboardType="numeric" placeholder="320.0" />

        <Text style={styles.label}>Stemming Length (m) [0.5 - 15.0m]</Text>
        <TextInput style={styles.input} value={stemming} onChangeText={setStemming} keyboardType="numeric" placeholder="5.0" />

        <TouchableOpacity style={styles.saveButton} onPress={handleSaveLog}>
          <Text style={styles.saveButtonText}>Save Field Log Offline</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 12, color: '#333' },
  card: { backgroundColor: '#fff', borderRadius: 8, padding: 16 },
  label: { fontSize: 13, fontWeight: '600', color: '#555', marginTop: 8 },
  input: { backgroundColor: '#f9f9f9', borderRadius: 6, padding: 10, borderWidth: 1, borderColor: '#ccc', marginTop: 4 },
  saveButton: { backgroundColor: '#1976D2', borderRadius: 6, padding: 14, alignItems: 'center', marginTop: 16 },
  saveButtonText: { color: '#fff', fontWeight: 'bold', fontSize: 14 },
});
