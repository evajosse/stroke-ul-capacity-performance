import numpy as np
import pandas as pd
from pathlib import Path
from plotting import population_trajectory

class FakeFit:
    def __init__(self):
        self.fe_params = pd.Series({'Intercept': 15., 'log1p_time_c': 12.})
    def cov_params(self):
        return pd.DataFrame([[.3,0],[0,.2]],index=self.fe_params.index,columns=self.fe_params.index)

class FakeSelection:
    def __init__(self):
        self.model = FakeFit()
        rows = []
        for i in range(3):
            for t in [21,51,111,201]:
                rows.append({'participant_id':i,'days_post_stroke':t,'arat':20+np.log1p((t-21)/30)*10})
        self.data=pd.DataFrame(rows)
        origin=self.data.groupby('participant_id')['days_post_stroke'].transform('min')
        lt=np.log1p((self.data.days_post_stroke-origin)/30)
        self.data['log1p_time_c']=lt-lt.mean()
        self.data['log1p_time2_c']=self.data.log1p_time_c**2
        self.data['time_c']=(self.data.days_post_stroke-self.data.days_post_stroke.mean())/30
        self.data['time_c2']=self.data.time_c**2

def test_plateau_outputs_and_reproducibility(tmp_path):
    obj=FakeSelection()
    population_trajectory(obj,outcome='arat',id_col='participant_id',time_col='days_post_stroke',output_dir=tmp_path,seed=9,n_parameter_draws=80)
    a=pd.read_csv(tmp_path/'population_plateau_summary.csv')
    d=pd.read_csv(tmp_path/'population_plateau_draws.csv')
    assert len(d)==80
    assert a.iloc[0].n_draws_with_crossing > 0
    assert a.iloc[0].plateau_fraction==.9
    before=d.copy()
    population_trajectory(obj,outcome='arat',id_col='participant_id',time_col='days_post_stroke',output_dir=tmp_path,seed=9,n_parameter_draws=80)
    pd.testing.assert_frame_equal(before,pd.read_csv(tmp_path/'population_plateau_draws.csv'))
