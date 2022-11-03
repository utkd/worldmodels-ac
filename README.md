## worldmodels-ac
## World Models using Actor Critic Controller

My (unsuccessful) attempt at implementing the World Models paper by Ha and Schmidhuber (https://arxiv.org/abs/1803.10122) using an Actor-Critic based controller instead of the original CMA-ES one

Disclaimer: The controller does not do a good job at solving the environment. I have not been able to fix it or identify the reason. 
I could only train the ActorCritic learner for upto 2000 episodes on my machine, which might probably not be enough.
I have stopped debugging the implementation for the time being and might retry in the future. 

### Running the Code
The command line arguments with the default values are shown. They can be omitted or changed as needed.
1. #### Generate rollouts

` python collect_initial_data.py --rollouts 100 --output saves/samples.npz `

2. #### Pretrain the VAE 

` python pretrain_vae.py --input saves/samples.npz --modelout saves/vae.mdl --output saves/zvalues.npz --batchsize 256 --epochs 10 `

3. #### Pretrain the MDN-RNN

` python pretrain_mdnrnn.py --input saves/zvalues.npz --modelout saves/mdnrnn.mdl --epochs 20 `

4. #### Train the Controller

` python train_controller.py --vaemodel saves/vae.mdl --rnnmodel saves/mdnrnn.mdl --modelout --saves/controller.mdl --episodes 1000 `

5. Run the trained Model

` python play.py --vaemodel saves/vae.mdl --rnnmodel saves/mdnrnn.mdl --controllermodel --saves/controller.mdl `
