#extract the winning genome from the pickle
import neat
import gym
import pickle
import numpy as np 
import retro
import cv2

env = retro.make('StreetFighter2SpecialChampionEdition', 'ken_onestar')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
	'config-feedforward')

#Data-visulization
population = neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

with open('winner_guile.pkl', 'rb') as input_file:
	genome = pickle.load(input_file)

ob = env.reset()
ac = env.action_space.sample()
inx, iny, inc = env.observation_space.shape
#the neural network
net = neat.nn.RecurrentNetwork.create(genome,config)

inx = int(inx/8)
iny = int(iny/8)

current_max_fitness = 0
fitness_current = 0
done = False
total_matches_won = 0
g_id = 1 #this is true only for the first iteration
health = 175
current_enemy_health = 176
g_id = 1
while not done :
#for i in range(0,1):
	env.render()

	ob = cv2.resize(ob,(inx,iny))
	ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
	ob = np.reshape(ob, (inx,iny))

	imgarray = np.ndarray.flatten(ob)

	nnOutput = net.activate(imgarray)

	ob, rew, done, info = env.step(nnOutput)

	np.delete(imgarray, 0)

	#reset health after a round
	if(info['enemy_health'] == 170):
		health = 170
	#add the fitness portion here --> Currently on matches_won and health
	current_matches_won = info['matches_won']
	if current_matches_won >= total_matches_won:
		fitness_current += 1000 * current_matches_won
		total_matches_won = current_matches_won
	if(info['enemy_health'] != 0):
		current_enemy_health = info['enemy_health']

	#change fitness based on damage dealt --> flawed, need to reset health after a round
	if current_enemy_health < health:
		fitness_current += health - current_enemy_health
		health = 176 - (176-current_enemy_health)

	#add penalty for losing health here

	#change the global fitness
	if fitness_current > current_max_fitness:
		current_max_fitness = fitness_current
	#print("The genome id is {0} and current fitness is {1}".format(genome_id, fitness_current))

	genome.fitness = fitness_current #fitness of the current genome


