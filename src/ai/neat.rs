use crate::ai::model::{NodeGene, ConnectionGene, NodeId, InnovationNumber, Evaluation, GenomeKey, Species};
use crate::ai::genome::Genome;
use std::collections::HashMap;
use rand::prelude::ThreadRng;
use crate::ai::calculator::Calculator;
use std::cmp::Ordering;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Parameters {
    pub asexual_reproduction_percent_chance: usize,
    pub mutate_link_percent_chance: usize,
    pub mutate_node_percent_chance: usize,
    pub mutate_link_tries: usize,
    pub mutate_enable_percent_chance: usize,
    pub reenable_percent_chance: usize,
    pub mutate_weight_percent_chance: usize,
    pub weight_shift_percent_chance: usize,
    pub weight_shift_range_min: f32,
    pub weight_shift_range_max: f32,
    pub weight_assign_range_min: f32,
    pub weight_assign_range_max: f32,
    pub compatibility_distance: f32,
    pub distance_numoff_excess_coefficient: f32,
    pub distance_numoff_disjoint_coefficient: f32,
    pub distance_average_weight_difference_coefficient: f32,
}

pub(crate) struct GlobalConnections{
    pub(crate) next_innovation: usize,
    pub(crate) connections: HashMap<(NodeId, NodeId), ConnectionGene>,
}

impl GlobalConnections{
    pub(crate) fn get_connection(&mut self, from: &NodeGene, to: &NodeGene) -> ConnectionGene{
        if let Some(connection) = self.connections.get(&(from.id(), to.id()))
            .map(|connection| connection.clone()){
            connection.clone()
        } else {
            let new_connection = ConnectionGene{
                id: InnovationNumber(self.next_innovation),
                from: from.clone(),
                to: to.clone()
            };
            self.next_innovation += 1;
            self.connections.insert((from.id(), to.id()), new_connection.clone());
            new_connection
        }
    }
}

pub struct Neat{
    connections: GlobalConnections,
    genomes: Vec<Genome>,
    input_nodes: usize,
    output_nodes: usize,
    params: Parameters,
    rng: ThreadRng,
    current_generation: usize,
}

impl Neat {
    pub fn init(input_nodes: usize, output_nodes: usize, initial_population: usize, params: Parameters) -> Self {
        let mut connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let mut genomes = Vec::new();
        let mut rng = rand::thread_rng();
        for i in 0 .. initial_population{
            let genome = Genome::init(input_nodes, output_nodes, &mut connections, &params, &mut rng);
            genomes.push(genome);
        }
        let current_generation = 0;

        Neat{connections, genomes, input_nodes, output_nodes, params, rng, current_generation}
    }

    pub fn genomes(&self) -> Vec<GenomeKey> {
        self.genomes.iter()
            .enumerate()
            .map(|(index, genome)| GenomeKey{index, generation: self.current_generation})
            .collect()
    }

    pub fn evolve(&mut self, evaluations: Vec<Evaluation>) {
        let wrong_generation = evaluations.iter()
            .filter(|evaluation| evaluation.genome.generation != self.current_generation)
            .count();
        if wrong_generation > 0 {
            panic!("Evaluation from wrong generation found");
        }
        let genome_count = evaluations.len();

        self.current_generation += 1;
        let mut species_list = self.place_genomes_into_species(evaluations);
        let mut next_generation_genomes = Vec::with_capacity(genome_count);
        let average_fitness: f32 = species_list.iter()
            .map(|species| species.genomes.iter()
                .map(|genome| genome.fitness)
                .fold(0f32,|acc,fitness| acc + fitness as f32))
            .fold(0f32, |acc, fitness| acc + fitness as f32) / genome_count as f32;
        for mut species in species_list.iter_mut(){
            //println!("evolve species {:?}", species);
            let mascot = self.genomes[species.mascot.index].clone();
            next_generation_genomes.push(mascot);
            let genome_count = species.genomes.len() as f32;
            let average_species_fitness = species.genomes.iter()
                .fold(0f32, |acc, genome| {
                    //println!("avg species calc: {} + {}", acc, genome.fitness);
                    acc + genome.fitness as f32
                }) / genome_count;
            println!("avg_species / avg = ? | {} / {} = {}", average_species_fitness, average_fitness, average_species_fitness / average_fitness);
            let target_genomes = ((average_species_fitness / average_fitness) * (genome_count)).round() as usize;
            println!("target genome count: {}, before: {}", target_genomes, genome_count);
            species.remove_least_fit();
            for _ in 1 .. target_genomes {
                //let evaluation = Self::select_random_genome(&species.genomes, &mut self.rng);
                let evaluation = &species.genomes[0];
                let genome = &self.genomes[evaluation.genome.index];
                if self.rng.gen_range(0, 100) < self.params.asexual_reproduction_percent_chance{
                    let new_genome = genome.mutate(&mut self.connections, &self.params, &mut self.rng);
                    next_generation_genomes.push(new_genome)
                } else {
                    let evaluation_partner = Self::select_random_genome(&species.genomes, &mut self.rng);
                    let partner = &self.genomes[evaluation_partner.genome.index];
                    let new_genome = if evaluation.fitness >= evaluation_partner.fitness {
                        genome.crossover(&partner, &mut self.rng)
                    } else {
                        partner.crossover(&genome, &mut self.rng)
                    };
                    let mutated = new_genome.mutate(&mut self.connections, &self.params, &mut self.rng);
                    next_generation_genomes.push(mutated);
                }
            }
        }
        println!("speciescount; {}", species_list.len());
        println!("evolve stop: {} genomes", next_generation_genomes.len());
        self.genomes = next_generation_genomes;
    }

    fn select_random_genome<'a>(genomes: &'a [Evaluation], rng: &mut impl Rng) -> &'a Evaluation{
        let count = genomes.len();
        let index = rng.gen_range(0, count);
        &genomes[index]
    }

    fn place_genomes_into_species(&mut self, mut evaluations: Vec<Evaluation>) -> Vec<Species> {
        let mut species_list: Vec<Species> = Vec::new();
        evaluations.sort_by(|a, b| a.fitness.cmp(&b.fitness).reverse());
        for evaluated in evaluations.into_iter(){
            let mut matching_species = None;
            for species in species_list.iter_mut(){
                let genome = &self.genomes[evaluated.genome.index];
                let mascot = &self.genomes[species.mascot.index];
                if genome.distance(mascot, &self.params) < self.params.compatibility_distance{
                    matching_species = Some(species);
                    break;
                }
            }
            match matching_species{
                None => {
                    let generation = self.current_generation;
                    let index = evaluated.genome.index;
                    let mut species = Species{ mascot: GenomeKey {index, generation}, genomes: vec![evaluated] };
                    species_list.push(species)
                },
                Some(species) => { species.genomes.push(evaluated) }
            }
        }
        species_list
    }

    pub fn decide(&self, genome_key: &GenomeKey, inputs: &[f32]) -> Vec<f32>{
        let genome = &self.genomes[genome_key.index];
        Calculator::calculate(inputs, genome)
    }

}