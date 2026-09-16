import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, SafeAreaView } from 'react-native';
import HomeScreen from './src/screens/HomeScreen';
import DesignsScreen from './src/screens/DesignsScreen';
import FieldLogScreen from './src/screens/FieldLogScreen';
import AlertsScreen from './src/screens/AlertsScreen';
import SyncScreen from './src/screens/SyncScreen';

export default function App() {
  const [currentTab, setCurrentTab] = useState('Home');
  const [userRole, setUserRole] = useState('blaster'); // Roles: 'blaster', 'engineer', 'supervisor'

  const renderScreen = () => {
    switch (currentTab) {
      case 'Home':
        return <HomeScreen role={userRole} onChangeRole={setUserRole} />;
      case 'Designs':
        return <DesignsScreen role={userRole} />;
      case 'FieldLog':
        return <FieldLogScreen role={userRole} />;
      case 'Alerts':
        return <AlertsScreen role={userRole} />;
      case 'Sync':
        return <SyncScreen role={userRole} />;
      default:
        return <HomeScreen role={userRole} onChangeRole={setUserRole} />;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>🇧🇼 BlastOpt Mobile</Text>
        <Text style={styles.headerSubtitle}>Role: {userRole.toUpperCase()}</Text>
      </View>

      <View style={styles.content}>{renderScreen()}</View>

      <View style={styles.navbar}>
        {['Home', 'Designs', 'FieldLog', 'Alerts', 'Sync'].map((tab) => (
          <TouchableOpacity
            key={tab}
            style={[styles.navButton, currentTab === tab && styles.navButtonActive]}
            onPress={() => setCurrentTab(tab)}
          >
            <Text style={[styles.navText, currentTab === tab && styles.navTextActive]}>
              {tab === 'FieldLog' ? 'Log' : tab}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 16,
    backgroundColor: '#1976D2',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  headerSubtitle: {
    color: '#e0e0e0',
    fontSize: 12,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  navbar: {
    flexDirection: 'row',
    backgroundColor: '#ffffff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  navButton: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
  },
  navButtonActive: {
    borderBottomWidth: 3,
    borderBottomColor: '#1976D2',
  },
  navText: {
    fontSize: 12,
    color: '#666666',
  },
  navTextActive: {
    color: '#1976D2',
    fontWeight: 'bold',
  },
});
