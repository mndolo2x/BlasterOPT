import React from 'react';
import { StyleSheet, Text, View, ScrollView } from 'react-native';

export default function AlertsScreen({ role }) {
  const alerts = [
    { id: 'ALT_001', type: 'VIBRATION_EXCEEDANCE', msg: 'PPV exceeded 10.0 mm/s limit at Pit Wall North (11.2 mm/s).', level: 'DANGER', time: '10:42 AM' },
    { id: 'ALT_002', type: 'GEOMETRY_DEVIATION', msg: 'Hole BH-108 overdrilled by 1.8m relative to design.', level: 'WARNING', time: '09:15 AM' },
    { id: 'ALT_003', type: 'AIRBLAST_RISK', msg: 'Stemming length in Hole BH-112 is < 3.0m (Airblast risk).', level: 'WARNING', time: '08:30 AM' },
  ];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Real-Time Field Alerts & Safety Warnings</Text>

      {alerts.map((alt) => (
        <View key={alt.id} style={[styles.card, alt.level === 'DANGER' ? styles.dangerCard : styles.warningCard]}>
          <View style={styles.headerRow}>
            <Text style={styles.alertType}>{alt.type}</Text>
            <Text style={styles.timeText}>{alt.time}</Text>
          </View>
          <Text style={styles.msgText}>{alt.msg}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 12, color: '#333' },
  card: { borderRadius: 8, padding: 14, marginBottom: 10, borderWidth: 1 },
  dangerCard: { backgroundColor: '#ffebee', borderColor: '#ef5350' },
  warningCard: { backgroundColor: '#fff3e0', borderColor: '#ffb74d' },
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  alertType: { fontWeight: 'bold', fontSize: 13, color: '#333' },
  timeText: { fontSize: 11, color: '#777' },
  msgText: { fontSize: 13, color: '#444' },
});
