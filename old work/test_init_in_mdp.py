from mdp import EpidemicMDP

infections = {'France': 0, 'Mauritius':1, 'South Africa': 1}
resources = 225
responses_csv = 'data/FR_MAUR_NIG_SA_responseIndicators.csv'
transitions_csv = 'data/FR_MAUR_NIG_SA_transitions.csv'

m = EpidemicMDP(transitions_csv, responses_csv, infections, resources)