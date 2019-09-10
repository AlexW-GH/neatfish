use quicksilver::graphics::Color;

#[derive(Clone, Debug, PartialEq)]
pub struct Player {
    pub flap_power: f32,
    pub pos_y: f32,
    pub width: f32,
    pub color: Color,
    pub alive: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Wall {
    pub top: f32,
    pub bot: f32,
    pub pos_x: f32,
    pub width: f32,
    pub color: Color,
}