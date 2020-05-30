import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# harga [0, 2000]
# rasa [0, 25]
# produksi [0, 100]

# define range
harga = ctrl.Antecedent(np.arange(0, 2000 + 1, 1), 'harga')
rasa = ctrl.Antecedent(np.arange(0, 25 + 1, 1), 'rasa')
produksi = ctrl.Consequent(np.arange(0, 100 + 1, 1), 'produksi')

harga['mahal'] = fuzz.trapmf(harga.universe, [1200, 1500, 2000, 2000])
harga['sedang'] = fuzz.trimf(harga.universe, [600, 1000, 1500])
harga['murah'] = fuzz.trapmf(harga.universe, [0, 0, 500, 800])

rasa['enak'] = fuzz.trapmf(rasa.universe, [10, 15, 25, 25])
rasa['kurang_enak'] = fuzz.trapmf(rasa.universe, [5, 8, 12, 15])
rasa['tidak_enak'] = fuzz.trapmf(rasa.universe, [0, 0, 7, 12])

produksi['besar'] = fuzz.trapmf(produksi.universe, [60, 75, 100, 100])
produksi['sedang'] = fuzz.trapmf(produksi.universe, [20, 25, 50, 75])
produksi['kecil'] = fuzz.trapmf(produksi.universe, [0, 10, 15, 25])

#  Rule
rule1 = ctrl.Rule(harga['sedang'] & rasa['enak'], produksi['besar'])
rule2 = ctrl.Rule(harga['murah'], produksi['besar'])
rule3 = ctrl.Rule(harga['sedang'] & rasa['tidak_enak'], produksi['sedang'])
rule4 = ctrl.Rule(harga['mahal'] & rasa['kurang_enak'], produksi['sedang'])

# control system
produksi_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

# simulation
jumlah_produksi = ctrl.ControlSystemSimulation(produksi_control)

# input value
# A(harga) -> 1400
# B(rasa) -> 15
jumlah_produksi.input['harga'] = 1400
jumlah_produksi.input['rasa'] = 15

jumlah_produksi.compute()
print(jumlah_produksi.output['produksi'])
produksi.view(sim=jumlah_produksi)