#Using the old population to train ken vs blanka

#a neural net learning to play the retro game street fighter
import gym
import retro
import argparse
import numpy as np
import cv2
import neat  #NeuroEvolutionOfAugmentedTechnologies is a genentic algorithm library
import pickle
import gzip
import random
#import visualize #will use it to see generational data

imgarray = []
#load the environment with the state
env = retro.make('StreetFighter2SpecialChampionEdition', 'ken_blanka')
env.mode = 'fast'

#dictionary storing genome_ids with the max amount of matches won-->give more fitness if it overperforms
genome_dict = {1:0}
#dictionary that holds the matches won by a genome in an iteration -> makes fitness more effec
temp_matches_won = {1:0}  





class Worker:
	
	def restore_Checkpoint(self,filename):
		"""Resumes the simulation from a previous saved point."""
		with gzip.open(filename) as f:
		    generation, config, population, species_set, rndstate = pickle.load(f)
		    random.setstate(rndstate)
		    return neat.Population(config, (population, species_set, generation))
	def eval_genome(self,genomes, config):
		#iterate through the genomes made
		for genome_id, genome in genomes:
			ob = env.reset()
			ac = env.action_space.sample()

			frame=0 #keep frame count
			xpos = 0 #note the x cordinate (how the char is moving)
			#these are the screen inputs: the x,y and the color gradients
			inx, iny, inc = env.observation_space.shape

			inx = int(inx/8)
			iny = int(iny/8)

			#print("the values are {0} and {1}".format(inx, iny))
			#the network -> will push inputs into the emulator
			net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
			current_max_fitness = 0
			fitness_current = 0
			done = False
			current_matches_won = 0
			g_id = 1 #this is true only for the first iteration
			health = 175
			current_enemy_health = 176
			our_health = 176
			our_health_flag = True
			info_health = 176
			while not done :
			#for i in range(0,1):
				#reduce the window size and change to grayscale
				frame+=1 
				#env.render()
				#genome based changes
				if g_id != genome_id:
					health = 175
					g_id = genome_id
					our_health = 176
					#store the new genome to the dictionary
					if  genome_id not in genome_dict:
						genome_dict[genome_id] = 0

					if genome_id not in temp_matches_won:
						genome_dict[genome_id] = 0

					temp_matches_won[genome_id] = 0

				ob = cv2.resize(ob,(inx,iny))
				#print the frame here
				#trying to convert the bit size of the image here
				ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)

				#reshape(input, newshape, order)
				ob = np.reshape(ob, (inx,iny))

				#x and y inputs to one array --> change to one dimension
				imgarray = np.ndarray.flatten(ob)
				#o/p from the neural network
				nnOutput = net.activate(imgarray)
				#pass the neural network output to the system
				ob, rew, done, info = env.step(nnOutput)
				#clear the image array for the next iteration
				np.delete(imgarray, 0)

				#reset health after a round
				if(info['enemy_health'] == 176):
					health = 176
				if our_health_flag is True:
					our_health = 176
					our_health_flag = False
				#add the fitness portion here --> Currently on matches_won and health
				
				#made a change here because max value of matches won is two
				current_matches_won += info['matches_won']
				if current_matches_won > genome_dict[genome_id]:
					#more fitness if it breaks previous record
					fitness_current += 1000 *current_matches_won
					genome_dict[genome_id] = current_matches_won
				elif current_matches_won > temp_matches_won[genome_id]:
					fitness_current += 1000
					temp_matches_won[genome_id] = current_matches_won

				if(info['enemy_health'] != 0):
					current_enemy_health = info['enemy_health']

				#change fitness based on damage dealt --> flawed, need to reset health after a round
				if current_enemy_health < health:
					fitness_current += health - current_enemy_health
					health = 176 - (176-current_enemy_health)

				'''if info['endgame-easy']:
					fitness_current += 100000
					done = True'''
				#add penalty for losing health here
				info_health = info['health']
				if info_health<=0:
					info_health = 176
				else:
					info_health = info['health']
				if info_health<our_health:
					fitness_current -= our_health - info_health
					our_health = info['health']

				#reset our health after a round
				if(our_health<=0) or info['health']<=0:
					our_health = 176
				#change the global fitness
				if fitness_current > current_max_fitness:
					current_max_fitness = fitness_current
				#print("The genome id is {0} and current fitness is {1}".format(genome_id, fitness_current))

				genome.fitness = fitness_current #fitness of the current genome


def main():
	work = Worker()
	population = work.restore_Checkpoint('neat-checkpoint')
	population.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	population.add_reporter(stats)
	population.add_reporter(neat.Checkpointer(10))

	winner = population.run(work.eval_genome,5)

	#dump the winner network with pickle
	with open('winner_nofeed.pkl', 'wb') as output:
		pickle.dump(winner,output,1)


if __name__ == '__main__':
	main()