use std::collections::{HashSet, HashMap};
use rand::Rng;
use crate::ai::model::{NodeGene, InnovationNumber, WeightedConnectionGene, NodeId, ConnectionGene, Weight};
use crate::ai::neat::{Neat, GlobalConnections, Parameters};

#[derive(Debug, Clone)]
pub struct Genome {
    nodes: Vec<NodeGene>,
    connections: Vec<WeightedConnectionGene>,
}

impl Genome {
    pub(crate) fn init(input_nodes: usize, output_nodes: usize, global_connections: &mut GlobalConnections, params: &Parameters, rng: &mut impl Rng) -> Self {
        let mut nodes = Vec::with_capacity(input_nodes + output_nodes + 1);
        let mut connections = Vec::new();

        nodes.push(NodeGene::BIAS(NodeId(0)));
        for i in 1 .. input_nodes + 1 {
            nodes.push(NodeGene::INPUT(NodeId(i)));
        }
        for i in input_nodes + 1 .. input_nodes + output_nodes + 1 {
            nodes.push(NodeGene::OUTPUT(NodeId(i)));
        }

        let mut genome = Genome{nodes, connections};

        while !genome.mutate_link(global_connections, params, rng){}

        genome
    }

    pub fn crossover(&self, recessive_genome: &Self, rng: &mut impl Rng) -> Self {
        let dominant_genes = &self.connections;
        let recessive_genes = &recessive_genome.connections;

        let mut crossover_connections = Vec::new();
        for i in 0 .. dominant_genes.len() {
            let dominant_gene = dominant_genes.get(i)
                .expect("Gene should be present");
            if let Some(recessive_gene) = recessive_genes.iter()
                .find(|gene|
                    gene.connection.id == dominant_gene.connection.id){
                let selected= pick_random(rng, dominant_gene, recessive_gene);
                crossover_connections.push(selected.clone())
            } else {
                crossover_connections.push(dominant_gene.clone())
            }
        }
        Genome{
            nodes: self.nodes.clone(),
            connections: crossover_connections,
        }
    }

    pub fn crossover_equal_fitness(&self, other_genome: &Self, rng: &mut impl Rng) -> Self {
        let self_connections = &self.connections;
        let other_connections = &other_genome.connections;

        let innovation_numbers: HashSet<InnovationNumber> = self.connections.iter()
            .map(|gene| gene.connection.id)
            .chain(other_connections.iter()
                .map(|gene| gene.connection.id))
            .collect();
        let mut sorted_numbers: Vec<InnovationNumber> = innovation_numbers.into_iter()
            .collect();
        sorted_numbers.sort();

        let mut crossover_connections = Vec::new();
        let mut crossover_nodes: HashSet<NodeGene> = HashSet::new();
        for innovation_number in sorted_numbers.iter() {
            let gene_self = self_connections.iter()
                .find(|gene| gene.connection.id == *innovation_number);
            let gene_other = other_connections.iter()
                .find(|gene| gene.connection.id == *innovation_number);
            if let Some(selected) = *pick_random(rng, &gene_self, &gene_other){
                crossover_connections.push(selected.clone());
                crossover_nodes.insert(selected.connection.from.clone());
                crossover_nodes.insert(selected.connection.to.clone());
            }
        }
        let required_nodes: Vec<&NodeGene> = self.nodes.iter()
            .filter(|gene| **gene != NodeGene::HIDDEN (gene.id()))
            .collect();
        for required_node in required_nodes {
            crossover_nodes.insert(required_node.clone());
        }
        let mut crossover_nodes: Vec<NodeGene> = crossover_nodes.into_iter().collect();
        crossover_nodes.sort_by(|a, b| a.id().value().cmp(&b.id().value()));

        Genome{
            nodes: crossover_nodes,
            connections: crossover_connections,
        }

    }

    pub(crate) fn mutate(&self, global_connections: &mut GlobalConnections, params: &Parameters, rng: &mut impl Rng) -> Self {
        let mut mutate = self.clone();

        if rng.gen_range(1, 100) >= params.mutate_link_percent_chance {
            let mut tries = 0;
            while mutate.mutate_link(global_connections, params, rng) && (tries < params.mutate_link_tries) {
                tries += 1;
            }
        }

        if rng.gen_range(1, 100) >= params.mutate_node_percent_chance {
            mutate.mutate_node(global_connections, rng);
        }

        if rng.gen_range(1, 100) >= params.mutate_weight_percent_chance {
            mutate.mutate_weight(params, rng);
        }

        if rng.gen_range(1, 100) >= params.mutate_enable_percent_chance {
            mutate.mutate_enabled(params, rng);
        }
        mutate
    }

    pub(crate) fn mutate_link(&mut self, global_connections: &mut GlobalConnections, params: &Parameters, rng: &mut impl Rng) -> bool {
        let valid_from_nodes = self.nodes.iter()
            .filter(|node| {
                match node {
                    NodeGene::BIAS(_) => true,
                    NodeGene::INPUT(_) => true,
                    NodeGene::OUTPUT(_) => false,
                    NodeGene::HIDDEN(_) => true,
                }})
            .collect::<Vec<&NodeGene>>();
        let valid_to_nodes = self.nodes.iter()
            .filter(|node| {
                match node {
                    NodeGene::BIAS(_) => false,
                    NodeGene::INPUT(_) => false,
                    NodeGene::OUTPUT(_) => true,
                    NodeGene::HIDDEN(_) => true
                }})
            .collect::<Vec<&NodeGene>>();

        if valid_to_nodes.len() > 0 {
            let node_from = valid_from_nodes[rng.gen_range(0, valid_from_nodes.len())];
            let node_to = valid_to_nodes[rng.gen_range(0, valid_to_nodes.len())];
            if Self::check_valid_connection(&self.connections, node_from, node_to) {
                let connection = global_connections.get_connection(node_from, node_to);
                let random_weight = rng.gen_range(
                    params.weight_assign_range_min,
                    params.weight_assign_range_max
                );
                self.connections.push(WeightedConnectionGene{
                    connection,
                    weight: Weight(random_weight),
                    enabled: true
                });
                true
            } else {
                false
            }
        } else {
            true
        }
    }

    fn check_valid_connection(existing_connections: &[WeightedConnectionGene], from: &NodeGene, to: &NodeGene) -> bool {
        let equal_count = existing_connections.iter()
            .filter(|weighted| weighted.connection.from == *from && weighted.connection.to == *to)
            .count();
        if equal_count > 0 {
            return false;
        }

        let mut nodes_used = vec![to.id()];
        Self::check_can_resolve_inputs(from.id(), existing_connections, &mut nodes_used)
    }

    fn check_can_resolve_inputs(node: NodeId, existing_connections: &[WeightedConnectionGene], nodes_used: &mut Vec<NodeId>) -> bool {
        if !nodes_used.contains(&node) {
            nodes_used.push(node);
            let inputs = existing_connections.iter()
                .filter(|weighted| weighted.connection.to.id() == node)
                .map(|weighted| weighted.connection.from.id());

            for input in inputs {
                if !Self::check_can_resolve_inputs(input, existing_connections, nodes_used){
                    return false
                }
            }
            true
        } else {
            false
        }

    }

    pub(crate) fn mutate_node(&mut self, global_connections: &mut GlobalConnections, rng: &mut impl Rng) {
        let selected_index: usize = rng.gen::<usize>() % self.connections.len();

        let mut gene_to_mutate = self.connections[selected_index].clone();
        let last_node = self.nodes.pop().expect("Nodes can not be empty");
        let new_node = NodeGene::HIDDEN(NodeId(last_node.id().value() + 1));
        let connection_to_new_node = global_connections.get_connection(&gene_to_mutate.connection.from, &new_node);
        let connection_from_new_node = global_connections.get_connection(&new_node, &gene_to_mutate.connection.to);

        self.connections.get_mut(selected_index).expect("").enabled = false;
        self.connections.push(WeightedConnectionGene{ connection: connection_to_new_node, weight: Weight(1f32), enabled: true });
        self.connections.push(WeightedConnectionGene{ connection: connection_from_new_node, weight: gene_to_mutate.weight, enabled: true });
        self.nodes.push(last_node);
        self.nodes.push(new_node);
    }

    fn mutate_weight(&mut self, params: &Parameters, rng: &mut impl Rng) {
        for mut connection in self.connections.iter_mut() {
            let set_shift_occurs: usize = rng.gen_range(0, 101);
            if set_shift_occurs <= params.weight_shift_percent_chance {
                let shift_value = rng.gen_range(
                    params.weight_shift_range_min,
                    params.weight_shift_range_max
                );
                connection.weight = connection.weight + shift_value;
            } else {
                let random_value = rng.gen_range(
                    params.weight_assign_range_min,
                    params.weight_assign_range_max
                );
                connection.weight = Weight(random_value)
            }
        }
    }

    fn mutate_enabled(&mut self, params: &Parameters, rng: &mut impl Rng) {
        let reenable_occurs: usize = rng.gen_range(0, 101);
        if reenable_occurs <= params.reenable_percent_chance {
            let mut disabled_connections: Vec<&WeightedConnectionGene> = self.connections.iter()
                .filter(|connection| !connection.enabled)
                .collect();
            if disabled_connections.len() > 0 {
                let chosen_connection = rng.gen_range(0, disabled_connections.len());
                let id = disabled_connections[chosen_connection].connection.id;
                self.connections.iter_mut()
                    .filter(|connection| connection.connection.id == id)
                    .for_each(|connection| connection.enabled = true)
            }
        } else {
            let connections_amount = self.connections.len();
            if connections_amount > 0 {
                let chosen_connection = rng.gen_range(0, connections_amount);
                self.connections[chosen_connection].toggle()
            }
        }
    }

    pub fn distance(&self, other_genome: &Self, params: &Parameters) -> f32 {
        let c1 = params.distance_numoff_excess_coefficient;
        let c2 = params.distance_numoff_disjoint_coefficient;
        let c3 = params.distance_average_weight_difference_coefficient;
        let n = 1f32;

        let (non_matching_left, non_matching_right) = Genome::get_non_matching_innovation_numbers(self, other_genome);
        let numoff_excess = Self::count_excess_connections(&non_matching_left, &non_matching_right) as f32;
        let numoff_disjoint = ((non_matching_left.len() + non_matching_right.len()) as f32) - numoff_excess;
        if numoff_disjoint < 0f32 {
            panic!("Disjoint genes cannot be less then 0");
        }
        let weight_difference = Self::calculate_average_weight_difference(self, other_genome);

        (c1*numoff_excess) / n + (c2*numoff_disjoint) / n + (c3+weight_difference) / n
    }

    fn get_non_matching_innovation_numbers(genome_left: &Genome, genome_right: &Genome) -> (Vec<InnovationNumber>, Vec<InnovationNumber>){
        let left_ids: Vec<InnovationNumber> = genome_left.connections.iter()
            .map(|connection| connection.connection.id)
            .collect();
        let right_ids: Vec<InnovationNumber> = genome_right.connections.iter()
            .map(|connection| connection.connection.id)
            .collect();

        let left_disjoint_connections = Genome::retrieve_non_matching_numbers(&left_ids, &right_ids);
        let right_disjoint_connections = Genome::retrieve_non_matching_numbers(&right_ids, &left_ids);

        (left_disjoint_connections, right_disjoint_connections)
    }

    fn retrieve_non_matching_numbers(innovation_numbers: &[InnovationNumber], compare_to: &[InnovationNumber]) -> Vec<InnovationNumber> {
        let mut disjoint_connections = Vec::new();
        for innovation_number in innovation_numbers.iter() {
            if !compare_to.contains(innovation_number) {
                disjoint_connections.push(innovation_number.clone())
            }
        }
        disjoint_connections
    }

    fn count_excess_connections(left: &[InnovationNumber], right: &[InnovationNumber]) -> usize {
        match left.last(){
            None => right.len(),
            Some(highest_left) => {
                match right.last(){
                    None => left.len(),
                    Some(highest_right) => {
                        if highest_left > highest_right {
                            left.iter()
                                .filter(|id| *id > highest_right)
                                .count()
                        } else if highest_right > highest_left {
                            right.iter()
                                .filter(|id| *id > highest_left)
                                .count()
                        } else {
                            panic!("Only non-matching innovation should have been provided")
                        }
                    }
                }
            }
        }
    }

    fn calculate_average_weight_difference(genome_left: &Genome, genome_right: &Genome) -> f32 {
        let accumulated_weights_left = genome_left.connections.iter()
            .map(|connection| connection.weight.value())
            .fold(0f32, |acc, x| acc + x );
        let average_weights_left = accumulated_weights_left / genome_left.connections.len() as f32;

        let accumulated_weights_right = genome_right.connections.iter()
            .map(|connection| connection.weight.value())
            .fold(0f32, |acc, x| acc + x );
        let average_weights_right = accumulated_weights_right / genome_right.connections.len() as f32;

        (average_weights_left - average_weights_right).abs()
    }

    pub(crate) fn connections(&self) -> &[WeightedConnectionGene] {
        &self.connections
    }

    pub(crate) fn nodes(&self) -> &[NodeGene] {
        &self.nodes
    }
}

fn pick_random<'a, T>(rng: &mut impl Rng, dominant: &'a T, recessive: &'a T) -> &'a T {
    let random: u32 = rng.gen();
    if random % 2 == 0 {
        dominant
    } else {
        recessive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::mock::StepRng;
    use crate::ai::model::{ConnectionGene, Weight};

    const TEST_PARAMETERS: Parameters = Parameters {
        mutate_link_tries: 0,
        mutate_enable_percent_chance: 0,
        reenable_percent_chance: 0,
        mutate_weight_percent_chance: 0,
        weight_shift_percent_chance: 0,
        weight_shift_range_min: 0.0,
        weight_shift_range_max: 0.0,
        weight_assign_range_min: 0.0,
        weight_assign_range_max: 0.0,
        compatibility_distance: 0.0,
        distance_numoff_excess_coefficient: 0.0,
        distance_numoff_disjoint_coefficient: 0.0,
        distance_average_weight_difference_coefficient: 0.0
    };

    #[test]
    fn crossover_similar_favour_dominant(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_similar_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 0);

        let result = dominant.crossover(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 2);
        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.1);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.1);
        assert_eq!(result.nodes.len(), 4);
    }

    #[test]
    fn crossover_similar_favour_recessive(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_similar_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(1, 0);

        let result = dominant.crossover(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 2);
        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.2);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.2);
        assert_eq!(result.nodes.len(), 4);
    }

    #[test]
    fn crossover_similar_favour_alternating(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_similar_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 1);

        let result = dominant.crossover(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 2);
        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.1);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.2);
        assert_eq!(result.nodes.len(), 4);
    }

    #[test]
    fn crossover_different_bigger_dominant(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_different_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 0);

        let result = dominant.crossover(&recessive, &mut fake_rng);

        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.1);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.1);
        assert_eq!(result.connections[2].connection.id.value(), 2);
        assert_eq!(result.connections[2].weight.value(), 0.1);
        assert_eq!(result.connections[3].connection.id.value(), 3);
        assert_eq!(result.connections[3].weight.value(), 0.1);
        assert_eq!(result.connections[4].connection.id.value(), 5);
        assert_eq!(result.connections[4].weight.value(), 0.1);
        assert_eq!(result.connections[5].connection.id.value(), 6);
        assert_eq!(result.connections[5].weight.value(), 0.1);
        assert_eq!(result.nodes.len(), 6);
    }

    #[test]
    fn crossover_different_smaller_dominant(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (recessive, dominant) = setup_crossover_different_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 0);

        let result = dominant.crossover(&recessive, &mut fake_rng);

        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.2);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.2);
        assert_eq!(result.connections[2].connection.id.value(), 3);
        assert_eq!(result.connections[2].weight.value(), 0.2);
        assert_eq!(result.connections[3].connection.id.value(), 4);
        assert_eq!(result.connections[3].weight.value(), 0.2);
        assert_eq!(result.nodes.len(), 5);
    }

    #[test]
    fn crossover_different_bigger_dominant_favour_alternating(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_different_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 1);

        let result = dominant.crossover(&recessive, &mut fake_rng);
        assert_eq!(result.connections[0].connection.id.value(), 0); //both, rng  : 0
        assert_eq!(result.connections[0].weight.value(), 0.1);
        assert_eq!(result.connections[1].connection.id.value(), 1); //both, rng  : 1
        assert_eq!(result.connections[1].weight.value(), 0.2);
        assert_eq!(result.connections[2].connection.id.value(), 2); //left, norng: 1
        assert_eq!(result.connections[2].weight.value(), 0.1);
        assert_eq!(result.connections[3].connection.id.value(), 3); //both, rng  : 0
        assert_eq!(result.connections[3].weight.value(), 0.1);
        assert_eq!(result.connections[4].connection.id.value(), 5); //left, norng: 0
        assert_eq!(result.connections[4].weight.value(), 0.1);
        assert_eq!(result.connections[5].connection.id.value(), 6); //left, norng: 0
        assert_eq!(result.connections[5].weight.value(), 0.1);
        assert_eq!(result.nodes.len(), 6);
    }

    #[test]
    fn crossover_different_smaller_dominant_favour_alternating(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (recessive, dominant) = setup_crossover_different_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 1);

        let result = dominant.crossover(&recessive, &mut fake_rng);

        assert_eq!(result.connections[0].connection.id.value(), 0); //both, rng  : 0
        assert_eq!(result.connections[0].weight.value(), 0.2);
        assert_eq!(result.connections[1].connection.id.value(), 1); //both, rng  : 1
        assert_eq!(result.connections[1].weight.value(), 0.1);
        assert_eq!(result.connections[2].connection.id.value(), 3); //both, rng  : 0
        assert_eq!(result.connections[2].weight.value(), 0.2);
        assert_eq!(result.connections[3].connection.id.value(), 4); //left, norng: 0
        assert_eq!(result.connections[3].weight.value(), 0.2);
        assert_eq!(result.nodes.len(), 5);
    }

    #[test]
    fn crossover_equal_fitness_similar_favour_first(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_similar_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 0);

        let result = dominant.crossover_equal_fitness(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 2);
        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.1);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.1);
        assert_eq!(result.nodes.len(), 4);
    }

    #[test]
    fn crossover_equal_fitness_similar_favour_second(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_similar_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(1, 0);

        let result = dominant.crossover_equal_fitness(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 2);
        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.2);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.2);
        assert_eq!(result.nodes.len(), 4);
    }

    #[test]
    fn crossover_equal_fitness_similar_favour_alternating(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_similar_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 1);

        let result = dominant.crossover_equal_fitness(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 2);
        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.1);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.2);
        assert_eq!(result.nodes.len(), 4);
    }

    #[test]
    fn crossover_equal_fitness_different_bigger_dominant(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_different_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 0);

        let result = dominant.crossover_equal_fitness(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 6);
        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.1);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.1);
        assert_eq!(result.connections[2].connection.id.value(), 2);
        assert_eq!(result.connections[2].weight.value(), 0.1);
        assert_eq!(result.connections[3].connection.id.value(), 3);
        assert_eq!(result.connections[3].weight.value(), 0.1);
        assert_eq!(result.connections[4].connection.id.value(), 5);
        assert_eq!(result.connections[4].weight.value(), 0.1);
        assert_eq!(result.connections[5].connection.id.value(), 6);
        assert_eq!(result.connections[5].weight.value(), 0.1);
        assert_eq!(result.nodes.len(), 6);
    }

    #[test]
    fn crossover_equal_fitness_different_smaller_dominant(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (recessive, dominant) = setup_crossover_different_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 0);

        let result = dominant.crossover_equal_fitness(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 4);
        assert_eq!(result.connections[0].connection.id.value(), 0);
        assert_eq!(result.connections[0].weight.value(), 0.2);
        assert_eq!(result.connections[1].connection.id.value(), 1);
        assert_eq!(result.connections[1].weight.value(), 0.2);
        assert_eq!(result.connections[2].connection.id.value(), 3);
        assert_eq!(result.connections[2].weight.value(), 0.2);
        assert_eq!(result.connections[3].connection.id.value(), 4);
        assert_eq!(result.connections[3].weight.value(), 0.2);
        assert_eq!(result.nodes.len(), 5);
    }

    #[test]
    fn crossover_equal_fitness_different_bigger_dominant_favour_alternating(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (dominant, recessive) = setup_crossover_different_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 1);

        let result = dominant.crossover_equal_fitness(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 5);
        assert_eq!(result.connections[0].connection.id.value(), 0); //0
        assert_eq!(result.connections[0].weight.value(), 0.1);
        assert_eq!(result.connections[1].connection.id.value(), 1); //1
        assert_eq!(result.connections[1].weight.value(), 0.2);
        assert_eq!(result.connections[2].connection.id.value(), 2); //0
        assert_eq!(result.connections[2].weight.value(), 0.1);
        assert_eq!(result.connections[3].connection.id.value(), 3); //1
        assert_eq!(result.connections[3].weight.value(), 0.2);
                                                                    //0
                                                                    //1
        assert_eq!(result.connections[4].connection.id.value(), 6); //0
        assert_eq!(result.connections[4].weight.value(), 0.1);
        assert_eq!(result.nodes.len(), 6);
    }

    #[test]
    fn crossover_equal_fitness_different_smaller_dominant_favour_alternating(){
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };
        let (recessive, dominant) = setup_crossover_different_genomes(&mut global_connections);
        let mut fake_rng = StepRng::new(0, 1);

        let result = dominant.crossover_equal_fitness(&recessive, &mut fake_rng);

        assert_eq!(result.connections.len(), 5);
        assert_eq!(result.connections[0].connection.id.value(), 0); //0
        assert_eq!(result.connections[0].weight.value(), 0.2);
        assert_eq!(result.connections[1].connection.id.value(), 1); //1
        assert_eq!(result.connections[1].weight.value(), 0.1);
                                                                    //0
        assert_eq!(result.connections[2].connection.id.value(), 3); //1
        assert_eq!(result.connections[2].weight.value(), 0.1);
        assert_eq!(result.connections[3].connection.id.value(), 4); //0
        assert_eq!(result.connections[3].weight.value(), 0.2);
        assert_eq!(result.connections[4].connection.id.value(), 5); //1
        assert_eq!(result.connections[4].weight.value(), 0.1);
        assert_eq!(result.nodes.len(), 6);
    }

    #[test]
    fn mutate_node_new_innovation(){
        let mut fake_rng = StepRng::new(1, 0);
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };

        let mut genome = setup_mutate_nodes_genome(&mut global_connections);
        genome.mutate_node(&mut global_connections, &mut fake_rng);

        assert_eq!(genome.connections.len(), 5);
        assert_eq!(genome.nodes.len(), 5);

        assert_connection(&genome, 0, 0.5, true, 0, 3, 0);
        assert_connection(&genome, 1, 0.5, false, 1, 3, 1);
        assert_connection(&genome, 2, 0.5, true, 2, 3, 2);
        assert_connection(&genome, 3, 1.0, true, 1, 4, 3);
        assert_connection(&genome, 4, 0.5, true, 4, 3, 4);
    }

    #[test]
    fn mutate_node_partial_new_innovation(){
        let mut fake_rng = StepRng::new(2, 0);
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };

        let mut genome = setup_mutate_nodes_genome(&mut global_connections);

        //Create existing ConnectionGenes in global_connections
        let node4 = NodeGene::HIDDEN(NodeId(4));
        let con14 = global_connections.get_connection(&genome.nodes[1], &node4);
        let con43 = global_connections.get_connection(&node4, &genome.nodes[3]);

        genome.mutate_node(&mut global_connections, &mut fake_rng);

        assert_eq!(genome.connections.len(), 5);
        assert_eq!(genome.nodes.len(), 5);

        assert_connection(&genome, 0, 0.5, true, 0, 3, 0);
        assert_connection(&genome, 1, 0.5, true, 1, 3, 1);
        assert_connection(&genome, 2, 0.5, false, 2, 3, 2);
        assert_connection(&genome, 3, 1.0, true, 2, 4, 5);
        assert_connection(&genome, 4, 0.5, true, 4, 3, 4);
    }

    #[test]
    fn mutate_node_existing_innovation(){
        let mut fake_rng = StepRng::new(1, 0);
        let mut global_connections = GlobalConnections{ next_innovation: 0, connections: HashMap::new() };

        let mut genome = setup_mutate_nodes_genome(&mut global_connections);

        //Create existing ConnectionGenes in global_connections
        let node4 = NodeGene::HIDDEN(NodeId(4));
        let con14 = global_connections.get_connection(&genome.nodes[1], &node4);
        let con43 = global_connections.get_connection(&node4, &genome.nodes[3]);

        genome.mutate_node(&mut global_connections, &mut fake_rng);

        assert_eq!(genome.connections.len(), 5);
        assert_eq!(genome.nodes.len(), 5);

        assert_connection(&genome, 0, 0.5, true, 0, 3, 0);
        assert_connection(&genome, 1, 0.5, false, 1, 3, 1);
        assert_connection(&genome, 2, 0.5, true, 2, 3, 2);
        assert_connection(&genome, 3, 1.0, true, 1, 4, 3);
        assert_connection(&genome, 4, 0.5, true, 4, 3, 4);
    }

    fn assert_connection(mutated: &Genome, index: usize, weight: f32, enabled: bool, from_id: usize, to_id: usize, innovation_id: usize) {
        assert_eq!(mutated.connections[index].weight.value(), weight);
        assert_eq!(mutated.connections[index].enabled, enabled);
        assert_eq!(mutated.connections[index].connection.from.id().value(), from_id);
        assert_eq!(mutated.connections[index].connection.to.id().value(), to_id);
        assert_eq!(mutated.connections[index].connection.id.value(), innovation_id);
    }

    fn setup_crossover_similar_genomes(global_connections: &mut GlobalConnections) -> (Genome, Genome){
        let node0 = NodeGene::BIAS (NodeId(0));
        let node1 = NodeGene::INPUT (NodeId(1));
        let node2 = NodeGene::INPUT (NodeId(2));
        let node3 = NodeGene::OUTPUT (NodeId(3));

        let left_con03 = global_connections.get_connection(&node0, &node3);
        let left_con13 = global_connections.get_connection(&node1, &node3);
        let left_connections = vec![
            WeightedConnectionGene { connection: left_con03, weight: Weight(0.1), enabled: true },
            WeightedConnectionGene { connection: left_con13, weight: Weight(0.1), enabled: true }
        ];
        let left = Genome{
            nodes: vec![node0, node1, node2, node3],
            connections: left_connections,
        };

        let right_con03 = global_connections.get_connection(&node0, &node3);
        let right_con13 = global_connections.get_connection(&node1, &node3);
        let right_connections = vec![
            WeightedConnectionGene{ connection: right_con03, weight: Weight(0.2), enabled: true },
            WeightedConnectionGene{ connection: right_con13, weight: Weight(0.2), enabled: true }
        ];
        let right = Genome{
            nodes: vec![node0, node1, node2, node3],
            connections: right_connections,
        };
        (left, right)
    }

    fn setup_crossover_different_genomes(global_connections: &mut GlobalConnections) -> (Genome, Genome){
        let (mut left, mut right) = setup_crossover_similar_genomes(global_connections);
        let node4 = NodeGene::HIDDEN(NodeId(4));
        let node5 = NodeGene::HIDDEN (NodeId(5));

        let left_con04 = global_connections.get_connection(&left.nodes[0], &node4);
        let left_con43 = global_connections.get_connection(&node4, &left.nodes[3]);
        left.nodes.push(node4);
        left.connections.push(WeightedConnectionGene { connection: left_con04, weight: Weight(0.1), enabled: true });
        left.connections.push(WeightedConnectionGene { connection: left_con43, weight: Weight(0.1), enabled: true });

        let right_con14 = global_connections.get_connection(&right.nodes[1], &node4);
        let right_con43 = global_connections.get_connection(&node4, &right.nodes[3]);
        right.nodes.push(node4);
        right.connections.push(WeightedConnectionGene { connection: right_con43, weight: Weight(0.2), enabled: true });
        right.connections.push(WeightedConnectionGene { connection: right_con14, weight: Weight(0.2), enabled: true });

        let left_con15 = global_connections.get_connection(&left.nodes[1], &node5);
        let left_con53 = global_connections.get_connection(&node5, &left.nodes[3]);
        left.nodes.push(node5);
        left.connections.push(WeightedConnectionGene { connection: left_con15, weight: Weight(0.1), enabled: true });
        left.connections.push(WeightedConnectionGene { connection: left_con53, weight: Weight(0.1), enabled: true });

        (left, right)
    }

    fn setup_mutate_nodes_genome(global_connections: &mut GlobalConnections) -> Genome {
        let node0 = NodeGene::BIAS(NodeId(0));
        let node1 = NodeGene::INPUT(NodeId(1));
        let node2 = NodeGene::INPUT(NodeId(2));
        let node3 = NodeGene::OUTPUT(NodeId(3));

        let con03 = global_connections.get_connection(&node0, &node3);
        let con13 = global_connections.get_connection(&node1, &node3);
        let con23 = global_connections.get_connection(&node2, &node3);

        let wcon03 = WeightedConnectionGene { connection: con03, weight: Weight(0.5), enabled: true };
        let wcon13 = WeightedConnectionGene { connection: con13, weight: Weight(0.5), enabled: true };
        let wcon23 = WeightedConnectionGene { connection: con23, weight: Weight(0.5), enabled: true };

        Genome {
            nodes: vec![node0, node1, node2, node3],
            connections: vec![wcon03, wcon13, wcon23],
        }
    }

}