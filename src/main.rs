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

        let rng = rand::thread_rng();
        let game_rules = GameRules {
            gravity: 4f32,
            flap_power: 14f32,
            flap_decay: 0.331f32,
            flap_delay: 0,
            max_bird_flap_power: 18f32,
            wall_spacing_min: 180,
            wall_spacing_max: 220,
            wall_width_min: 30,
            wall_width_max: 65,
            wall_hole_size_min: 80,
            wall_hole_size_max: 130,
            wall_hole_pos_top_min: 20,
            wall_hole_pos_top_max: MAP_SIZE_PX_Y - 135
        };

        let game_size = Vector::new(MAP_SIZE_PX_X as f32, MAP_SIZE_PX_Y as f32);

        let neat_bird = NeatBird::new(game_size, game_rules, 50, rng);

        let game_window = GameWindow{};
        let control_window = ControlWindow{};
        let info_window = InfoWindow{};
        let neat_params = Parameters{
            mutate_link_tries: 0,
            enable_mutate_percent_chance: 0,
            reenable_percent_chance: 0,
            weight_mutate_percent_chance: 0,
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
        let neat = Neat::init(4, 1, 50, neat_params);


        Ok(Self { neat_bird, game_window, control_window, info_window, neat})
    }

    fn update(&mut self, window: &mut Window) -> Result<()> {
        use quicksilver::input::ButtonState::Pressed;
        let brains = self.neat.genomes();
        let decisions: Vec<bool> = brains.iter().map(|brain| {
            let inputs = vec![1f32, 0f32, 1f32, 0f32];
            self.neat.decide(brain, &inputs)
        }).map(|output| {
            if output[0] > 0.5f32{
                true
            } else {
                false
            }
        }).collect();
        let keys_pressed = vec![
            window.keyboard()[Key::Space] == Pressed,
            window.keyboard()[Key::LControl] == Pressed
        ];
        self.neat_bird.update_frame(&decisions);

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
        let players_alive = players.iter().filter(|player| player.alive).count() as f32;
        players.iter().filter(|player| player.alive).for_each(|player|
            window.draw(
                &Rectangle::new((1f32, player.pos_y), (player.width, player.width)),
                Col(player.color.with_alpha(0.075f32.max(1f32 / players_alive)))
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
