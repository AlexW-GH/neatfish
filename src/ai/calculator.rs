use crate::ai::genome::Genome;
use crate::ai::neat::GlobalConnections;
use crate::ai::model::{NodeId, NodeGene};

pub struct Calculator;

impl Calculator{
    pub fn calculate(inputs: &[f32], genome: &Genome) -> Vec<f32>{
        let mut calc_nodes = Self::generate_calculation_nodes(inputs, genome);
        let mut output_nodes: Vec<CalcGene> = calc_nodes.iter()
            .filter(|node| match node.node {
                NodeGene::OUTPUT(_) => true,
                _ => false,
            })
            .map(|node| node.clone())
            .collect();

        for mut output in output_nodes.iter_mut(){
            let result = Self::calculate_internal(output.clone(), &mut calc_nodes);
            output.value = Some(result);
        }

        output_nodes.iter()
            .map(|node| node.value.expect("Value should have been already calculated"))
            .collect()
    }

    fn calculate_internal(node: CalcGene, all_nodes: &mut [CalcGene]) -> f32 {
        let mut result = 0f32;
        for input in node.inputs.iter() {
            let input_value = all_nodes[input.0.value()].value;
            if let Some(value) = input_value {
                result += value;
            } else {
                let input_node = all_nodes[input.0.value()].clone();
                let calculation = Self::calculate_internal(input_node, all_nodes);
                all_nodes[input.0.value()].value = Some(calculation);
                result += calculation;
            }
        }
        result
    }

    fn generate_calculation_nodes(inputs: &[f32], genome: &Genome) -> Vec<CalcGene> {
        let nodes = genome.nodes();
        let connections = genome.connections();

        nodes.iter().enumerate()
            .map(|(index, node)| {
                let node_id = node.id();
                let value = match node {
                    NodeGene::BIAS(_) => Some(1f32),
                    NodeGene::INPUT(_) => Some(inputs[index-1]),
                    NodeGene::OUTPUT(_) => None,
                    NodeGene::HIDDEN(_) => None
                };
                let inputs = connections.iter()
                    .filter(|weighted| weighted.connection.to.id() == node_id)
                    .map(|weighted| (weighted.connection.from.id(), weighted.weight.value()))
                    .collect();

                CalcGene {
                    node: node.clone(),
                    value,
                    inputs,
                }})
            .collect()
    }
}

#[derive(Clone)]
struct CalcGene {
    node: NodeGene,
    value: Option<f32>,
    inputs: Vec<(NodeId, f32)>,
}