pub mod neat_bird;
pub mod model;

trait MinMax{
    type T;

    fn min_max(self, min: Self::T, max: Self::T) -> Self::T;
}

impl MinMax for f32{
    type T = f32;

    fn min_max(self, min: f32, max: f32) -> f32 {
        let value = f32::min(self, max);
        f32::max(value, min)
    }
}