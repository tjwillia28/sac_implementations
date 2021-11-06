### Implementations of Soft Actor-Critic with fixed temperature(SAC1) and Soft Actor-Critic with learned temperature(SAC2)
- SAC1: sac_fixed_temp.py
- SAC2: sac_learned_temp.py
- Actor NN architecture: actor.py
- Critic NN architecture: critic.py
- sac2_scratch_main_class_fixed_temp.ipynb shows training curve of SAC1
- sac2_scratch_main_class_learned_temp.ipynb shows training curve of SAC2

### Implementations of SAC2 where NN architectures use Batch Normalization and Weight Decay
- SAC2 with BN + L2 reg.: sac2_scratch_main_class_learned_temp_batchnorm_weightdecay.ipynb , notebook also shows training curve
- Actor NN with BN + L2 reg.: actor_bn_l2.py
- Critic NN with BN + L2 reg.: critic_bn_l2.py

#### Dependencies
- gym==0.17.2
- keras==2.5.0
- matplotlib==3.1.1
- numpy==1.19.5
- tensorflow==2.5.0
- tensorflow-probability==0.12.2