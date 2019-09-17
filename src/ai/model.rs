use std::ops::Add;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InnovationNumber(pub(crate) usize);

impl InnovationNumber{
    pub fn value(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    pub fn value(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weight(pub(crate) f32);

impl Weight{
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Add<f32> for Weight {
    type Output = Weight;

    fn add(self, rhs: f32) -> Self::Output {
        Weight(self.0 + rhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NodeGene {
    HIDDEN(NodeId),
    INPUT(NodeId),
    OUTPUT(NodeId),
    BIAS(NodeId),
}

impl NodeGene {
    pub fn id(&self) -> NodeId {
        match *self {
            NodeGene::HIDDEN (id) => id,
            NodeGene::INPUT (id) => id,
            NodeGene::OUTPUT (id) => id,
            NodeGene::BIAS (id) => id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConnectionGene {
    pub id: InnovationNumber,
    pub from: NodeGene,
    pub to: NodeGene,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedConnectionGene {
    pub connection: ConnectionGene,
    pub weight: Weight,
    pub enabled: bool,
}

impl WeightedConnectionGene {
    pub fn toggle(&mut self){
        self.enabled = !self.enabled;
    }
}

#[derive(Debug, Clone)]
pub struct GenomeKey{
    pub(crate) index: usize,
    pub(crate) generation: usize,
}

#[derive(Debug, Clone)]
pub struct Evaluation{
    pub genome: GenomeKey,
    pub fitness: usize,
}

#[derive(Debug)]
pub(crate) struct Species{
    pub(crate) mascot: GenomeKey,
    pub(crate) genomes: Vec<Evaluation>,
}

impl Species{
    pub(crate) fn remove_least_fit(&mut self){
        self.genomes.sort_by(|left, right| left.fitness.cmp(&right.fitness));
        if self.genomes.len() > 2 {
            let keep = self.genomes.len() / 2;
            self.genomes = self.genomes.iter().enumerate()
                .filter(|(index, eval)| *index < keep )
                .map(|(index, eval)| eval.clone())
                .collect();
        }

    }
}
