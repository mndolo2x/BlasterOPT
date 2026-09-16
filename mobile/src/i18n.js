/**
 * Internationalization (i18n) Module for BlastOpt Mobile (React Native).
 *
 * English and Setswana translation dictionary and translation helper function.
 */

export const MOBILE_TRANSLATIONS = {
  en: {
    app_title: 'BlastOpt Mobile',
    role_blaster: 'BLASTER',
    role_engineer: 'ENGINEER',
    role_supervisor: 'SUPERVISOR',
    nav_home: 'Home',
    nav_designs: 'Designs',
    nav_field_log: 'Log',
    nav_alerts: 'Alerts',
    nav_sync: 'Sync',
    title_field_log: 'Field Data Log Entry (Offline First)',
    label_hole_id: 'Hole ID (e.g. BH-101)',
    label_depth: 'Measured Depth (m)',
    label_charge_weight: 'Charge Weight (kg)',
    label_stemming: 'Stemming Length (m)',
    btn_save_offline: 'Save Field Log Offline',
    btn_sync_queue: 'Replay & Reconcile Sync Queue',
  },
  tn: {
    app_title: 'BlastOpt Mobile (Setswana)',
    role_blaster: 'MOTHUNYI',
    role_engineer: 'MOWENJINERE',
    role_supervisor: 'MOLAODI',
    nav_home: 'Motheo',
    nav_designs: 'Dipolane',
    nav_field_log: 'Kwala',
    nav_alerts: 'Ditsiboso',
    nav_sync: 'Sync',
    title_field_log: 'Sekwalo sa Ditlhaka tsa Mmu (Offline First)',
    label_hole_id: 'Lekwalo la Mosima (BH-101)',
    label_depth: 'Botelello ba Mosima (m)',
    label_charge_weight: 'Boremelelo ba Moko (kg)',
    label_stemming: 'Sebaka sa Stemming (m)',
    btn_save_offline: 'Boloka Sekwalo le fa go se Network',
    btn_sync_queue: 'Romela Ditlhaka tse di Bolokilweng',
  },
};

/**
 * Returns localized string for mobile app key.
 *
 * @param {string} key - Translation key.
 * @param {string} lang - Language code ('en' or 'tn').
 * @returns {string} Localized string.
 */
export function t(key, lang = 'en') {
  const dict = MOBILE_TRANSLATIONS[lang] || MOBILE_TRANSLATIONS.en;
  return dict[key] || MOBILE_TRANSLATIONS.en[key] || key;
}
