mod game;
mod gui;
mod ai;

use quicksilver::{
    geom::{Vector, Rectangle},
    graphics::{Color, Background::Col},
    lifecycle::{run, Settings, State, Window},
    input::Key,
    Result,
};
use crate::ai::neat::{Neat, Parameters};
use crate::game::neat_bird::{NeatBird, GameRules};
use crate::gui::{GameWindow, InfoWindow, ControlWindow};
use quicksilver::geom::Line;
use quicksilver::graphics::BlendMode;
use crate::ai::Evaluation;
use rand::rngs::StdRng;
use rand::SeedableRng;

const MAP_SIZE_PX_X: usize = 800;
const MAP_SIZE_PX_Y: usize = 600;

struct NeatGame {
    neat_bird: NeatBird,
    game_window: GameWindow,
    info_window: InfoWindow,
    control_window: ControlWindow,
    neat: Neat,
}

impl State for NeatGame {
    fn new() -> Result<Self> {

        let neat_params = Parameters{
            asexual_reproduction_percent_chance: 5,
            mutate_link_percent_chance: 10,
            mutate_node_percent_chance: 2,
            mutate_enable_percent_chance: 5,
            mutate_weight_percent_chance: 15,
            reenable_percent_chance: 75,
            weight_shift_percent_chance: 85,
            mutate_link_tries: 3,
            weight_shift_range_min: -0.2,
            weight_shift_range_max: 0.2,
            weight_assign_range_min: -3.0,
            weight_assign_range_max: 3.0,
            compatibility_distance: 10.0,
            distance_numoff_excess_coefficient: 1.0,
            distance_numoff_disjoint_coefficient: 1.0,
            distance_average_weight_difference_coefficient: 1.7
        };
        let neat = Neat::init(4, 1, 1000, neat_params);

        let game_rules = GameRules {
            gravity: 4f32,
            flap_power: 14f32,
            flap_decay: 0.331f32,
            flap_delay: 0,
            max_bird_flap_power: 18f32,
            wall_spacing_min: 180,
            wall_spacing_max: 200,
            wall_width_min: 30,
            wall_width_max: 65,
            wall_hole_size_min: 100,
            wall_hole_size_max: 120,
            wall_hole_pos_top_min: 20,
            wall_hole_pos_top_max: MAP_SIZE_PX_Y - 140
        };

        let game_size = Vector::new(MAP_SIZE_PX_X as f32, MAP_SIZE_PX_Y as f32);

        let neat_bird = NeatBird::new(game_size, game_rules, &neat.genomes(), "TechTalk");

        let game_window = GameWindow{};
        let control_window = ControlWindow{};
        let info_window = InfoWindow{};

        Ok(Self { neat_bird, game_window, control_window, info_window, neat})
    }

    fn update(&mut self, window: &mut Window) -> Result<()> {
        use quicksilver::input::ButtonState::Pressed;
        let decisions: Vec<bool> = {
            let players = self.neat_bird.players();
            let next_wall = self.neat_bird.next_wall();
            let pos_x = self.neat_bird.covered_distance();
            let (wall_x, wall_top, wall_bot) = next_wall.map(|wall| {
                let wall_distance = (wall.pos_x - pos_x as f32) / MAP_SIZE_PX_X as f32;
                let wall_top = wall.top as f32 / MAP_SIZE_PX_Y as f32;
                let wall_bot = wall.bot as f32 / MAP_SIZE_PX_Y as f32;
                (wall_distance, wall_top, wall_bot)
            }).or(Some((0f32, 0f32, 0f32))).unwrap();

            players.iter().map(|(player, evaluation)| {
                let player_y =  player.pos_y / MAP_SIZE_PX_Y as f32;
                let flap_power =  player.flap_power;

                //println!("player_y: {}, wall_x: {}, wall_top: {}, wall_bot: {}", player_y, wall_x, wall_top, wall_bot);
                let inputs = vec![player_y, wall_x, wall_top, wall_bot];
                self.neat.decide(&evaluation.genome, &inputs)
            }).map(|output| {
                if output.len() > 0 && output[0] > 0.5f32{
                    true
                } else {
                    false
                }
            }).collect()
        };
        if self.neat_bird.update_frame(&decisions) || self.neat_bird.covered_distance() >= 4000 {
            let players = self.neat_bird.players();
            let evaluations = players.iter()
                .map(|(player, evaluation)| {
                    if player.alive {
                        let mut evaluation = evaluation.clone();
                        evaluation.fitness = 4001;
                        evaluation
                    } else {
                        evaluation.clone()
                    }
                } )
                .collect();
            self.neat.evolve(evaluations);
            self.neat_bird.reset();
            let genomes = self.neat.genomes();
            self.neat_bird.set_players(&genomes);
        }
        Ok(())
    }

    fn draw(&mut self, window: &mut Window) -> Result<()> {
        window.clear(Color::WHITE)?;

        let walls = self.neat_bird.walls();
        let game_size = self.neat_bird.game_size();
        let covered_distance = self.neat_bird.covered_distance();
        window.draw(
            &Line::new((0f32, 0f32), (game_size.x, 0f32)),
            Col(Color::RED)
        );
        window.draw(
            &Line::new((0f32, game_size.y), (game_size.x, game_size.y)),
            Col(Color::RED)
        );

        for wall in walls {
            let in_screen_width = if wall.pos_x - covered_distance as f32 + wall.width > game_size.x {
                game_size.x - (wall.pos_x - covered_distance as f32)
            } else {
                wall.width
            };
            window.draw(
                &Rectangle::new((wall.pos_x - covered_distance as f32, 0), (in_screen_width, wall.top)),
                Col(wall.color)
            );
            window.draw(
                &Rectangle::new((wall.pos_x - covered_distance as f32, game_size.y), (in_screen_width, -wall.bot )),
                Col(wall.color)
            );
        }

        let players = self.neat_bird.players();
        let players_alive = players.iter().filter(|player| player.0.alive).count() as f32;
        players.iter().filter(|player| player.0.alive).for_each(|player|
            window.draw(
                &Rectangle::new((1f32, player.0.pos_y), (player.0.width, player.0.width)),
                Col(player.0.color.with_alpha(0.075f32.max(1f32 / players_alive)))
            )
        );
        Ok(())
    }
}

fn main() {
    let settings = Settings {
        ..Default::default()
    };
    run::<NeatGame>("NEATFish", Vector::new(1600, 800), settings);
}
