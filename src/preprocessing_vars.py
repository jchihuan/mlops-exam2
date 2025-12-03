import numpy as np
import pandas as pd

# ==================
# PROCESAR VARIABLES
# ==================
def one_hot_encoding(df, col, categories=None):
    selected_col = df[col]

    if categories is not None: # Forzar las categorias del one hot encoding
        selected_col = selected_col.astype(pd.CategoricalDtype(categories))

    new_cols = pd.get_dummies(selected_col, prefix=col)
    df = df.drop([col], axis='columns')
    df = pd.concat([df, new_cols], axis='columns')

    return df, new_cols

def process_vars(df):
    variables_categoricas = ['ent_1erlntcrallsfm01']
    variables_numericas = ['nro_producto_6m', 'prom_uso_tc_rccsf3m', 'ctd_sms_received',
                           'max_usotcribksf06m', 'ctd_camptot06m', 'dsv_svppallsf06m',
                           'prm_svprmecs06m', 'ctd_app_productos_m1', 'ctd_campecsm01',
                           'lin_tcrrstsf03m', 'mnt_ptm', 'dif_no_gestionado_4meses',
                           'max_campecs06m', 'beta_pctusotcr12m', 'rat_disefepnm01',
                           'flg_saltotppe12m', 'prom_sow_lintcribksf3m', 'openhtml_1m',
                           'nprod_1m', 'nro_transfer_6m', 'max_usotcrrstsf03m',
                           'prm_cnt_fee_amt_u7d', 'pas_avg6m_max12m', 'beta_saltotppe12m',
                           'seg_un', 'ant_ultprdallsf', 'avg_sald_pas_3m', 'pas_1m_avg3m',
                           'num_incrsaldispefe06m', 'cnl_age_p4m_p12m', 'cnl_atm_p4m_p12m',
                           'cre_lin_tc_rccibk_m07', 'prm_svprmlibdis06m', 'ingreso_neto',
                           'max_nact_12m', 'cre_sldtotfinprm03', 'dif_contacto_efectivo_10meses',
                           'act_1m_avg3m', 'monto_consumos_ecommerce_tc', 'ctd_camptotm01',
                           'prop_atm_4m', 'prom_pct_saldopprcc6m', 'apppag_1m', 'nro_configuracion_6m',
                           'act_avg6m_max12m', 'sldvig_tcrsrcf', 'prom_score_acepta_12meses',
                           'telefonos_6meses', 'pas_1m_avg6m', 'ctd_camptototrcnl06m',
                           'prm_saltotrdpj03m', 'bpitrx_1m', 'prm_lintcribksf03m', 'ctd_entrdm01',
                           'avg_openhtml_6m', 'tea', 'pct_usotcrm01','senthtml_1m']

    df = df.replace(['', 'null', 'None'], [np.nan, np.nan, np.nan])

    for column in variables_numericas:
        df[column] = df[column].fillna(-9999999)

    df = df.astype({v: 'float32' for v in variables_numericas})
    df = df.astype({v: 'string' for v in variables_categoricas})
    df = df.astype({v: 'string' for v in ['partition']})

    Variables_faltanLlenarNulos = {'ent_1erlntcrallsfm01': ['INTERBANK']}

    for a in Variables_faltanLlenarNulos:
        df[a] = df[a].fillna('SV')
        df.loc[~df[a].isin(Variables_faltanLlenarNulos[a]), a] = 'OTRO'

    cols_dumm = pd.DataFrame()
    for col in Variables_faltanLlenarNulos:
        default_value = 'OTRO'
        values = Variables_faltanLlenarNulos[col]
        values.append(default_value)
        df, dummy = one_hot_encoding(df, col, values)
        cols_dumm = pd.concat([cols_dumm, dummy], axis='columns')

    return df