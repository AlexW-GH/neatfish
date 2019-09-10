use crate::ai::model::{NodeGene, ConnectionGene, NodeId, InnovationNumber, Evaluation, GenomeKey, Species};
use crate::ai::genome::Genome;
use std::collections::HashMap;
use rand::prelude::ThreadRng;
use crate::ai::calculator::Calculator;

#[derive(Debug, Clone)]
pub struct Parameters {
    pub mutate_link_tries: usize,
    pub enable_mutate_percent_chance: usize,
    pub reenable_percent_chance: usize,
    pub weight_mutate_percent_chance: usize,
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
    next_innovation: usize,
    connections: HashMap<(NodeId, NodeId), ConnectionGene>,
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
            let genome = Genome::init(input_nodes, output_nodes, &mut connections, &mut rng);
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

        self.current_generation += 1;
        let species_list = self.place_genomes_into_species(evaluations);

        let mut mutations = Vec::new();
        for genome in self.genomes.iter(){
            let mutated = genome.mutate(&mut self.connections, &self.params, &mut self.rng);
            mutations.push(mutated);
        }
        unimplemented!()
    }

    fn place_genomes_into_species(&mut self, evaluations: Vec<Evaluation>) -> Vec<Species> {
        let mut species_list: Vec<Species> = Vec::new();
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