import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, ScrollView } from 'react-native';

export default function DesignsScreen({ role }) {
  const [search, setSearch] = useState('');

  const designs = [
    { id: 'DES_JWA_001', name: 'Jwaneng Cut 8 Bench 15S', holes: 24, pf: 0.65, status: 'APPROVED' },
    { id: 'DES_ORA_002', name: 'Orapa Pit 1 Bench 12N', holes: 36, pf: 0.72, status: 'APPROVED' },
    { id: 'DES_KAR_003', name: 'Karowe South Kimberlite', holes: 18, pf: 0.58, status: 'DRAFT' },
  ];

  const filtered = designs.filter(d => d.name.toLowerCase().includes(search.toLowerCase()) || d.id.toLowerCase().includes(search.toLowerCase()));

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Blast Designs List</Text>

      <TextInput
        style={styles.searchInput}
        placeholder="Search design by ID or name..."
        value={search}
        onChangeText={setSearch}
      />

      {filtered.map((item) => (
        <View key={item.id} style={styles.card}>
          <Text style={styles.cardHeader}>{item.id}: {item.name}</Text>
          <Text style={styles.cardText}>Holes: {item.holes} | Powder Factor: {item.pf} kg/m³</Text>
          <Text style={[styles.statusText, item.status === 'APPROVED' ? styles.approved : styles.draft]}>
            Status: {item.status}
          </Text>
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 12, color: '#333' },
  searchInput: { backgroundColor: '#fff', borderRadius: 8, padding: 12, marginBottom: 12, borderWidth: 1, borderColor: '#ccc' },
  card: { backgroundColor: '#fff', borderRadius: 8, padding: 16, marginBottom: 10 },
  cardHeader: { fontSize: 16, fontWeight: 'bold', color: '#1976D2' },
  cardText: { fontSize: 14, color: '#555', marginTop: 4 },
  statusText: { fontSize: 12, fontWeight: 'bold', marginTop: 6 },
  approved: { color: '#2e7d32' },
  draft: { color: '#f57c00' },
});
