import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView } from 'react-native';

export default function HomeScreen({ role, onChangeRole }) {
  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Field Dashboard & Shift Summary</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Shift Overview</Text>
        <Text style={styles.cardText}>Mine Site: Jwaneng Cut 8</Text>
        <Text style={styles.cardText}>Active Bench: BENCH_15_SOUTH</Text>
        <Text style={styles.cardText}>Drilled Holes Logged: 18 / 24</Text>
        <Text style={styles.cardText}>Pending Sync Queue: 3 items</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Role-Based Access Control (RBAC)</Text>
        <Text style={styles.cardText}>Select Active User Role:</Text>
        <View style={styles.roleContainer}>
          {['blaster', 'engineer', 'supervisor'].map((r) => (
            <TouchableOpacity
              key={r}
              style={[styles.roleButton, role === r && styles.roleButtonActive]}
              onPress={() => onChangeRole(r)}
            >
              <Text style={[styles.roleText, role === r && styles.roleTextActive]}>{r.toUpperCase()}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recent Field Alerts</Text>
        <Text style={[styles.alertText, styles.warningText]}>⚠️ Hole BH-104 Depth Deviation (+1.2m)</Text>
        <Text style={[styles.alertText, styles.dangerText]}>🚨 PPV Sensor Near Pit Wall Triggered (8.8 mm/s)</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 12, color: '#333' },
  card: { backgroundColor: '#fff', borderRadius: 8, padding: 16, marginBottom: 12, elevation: 2 },
  cardTitle: { fontSize: 16, fontWeight: 'bold', marginBottom: 8, color: '#1976D2' },
  cardText: { fontSize: 14, color: '#555', marginBottom: 4 },
  roleContainer: { flexDirection: 'row', marginTop: 8 },
  roleButton: { flex: 1, padding: 8, marginHorizontal: 2, borderRadius: 4, borderWidth: 1, borderColor: '#ccc', alignItems: 'center' },
  roleButtonActive: { backgroundColor: '#1976D2', borderColor: '#1976D2' },
  roleText: { fontSize: 12, color: '#333' },
  roleTextActive: { color: '#fff', fontWeight: 'bold' },
  alertText: { fontSize: 13, marginBottom: 4 },
  warningText: { color: '#e65100' },
  dangerText: { color: '#c62828' },
});
