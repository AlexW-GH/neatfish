use quicksilver::geom::{Vector, Rectangle};
use crate::game::model::{Player, Wall};
use quicksilver::graphics::Color;
use crate::game::MinMax;
use std::collections::vec_deque::VecDeque;
use rand::Rng;
use rand::SeedableRng;
use std::cell::RefCell;
use rand::rngs::StdRng;
use crate::ai::{Evaluation, GenomeKey};

pub struct NeatBird {
    walls: VecDeque<Wall>,
    players: Vec<(Player, Evaluation)>,
    game_size: Vector,
    covered_distance: usize,
    rules: GameRules,
    next_wall_spacing: usize,
    wall_spacing_counter: usize,
    rng: StdRng,
    seed: [u8; 32],
}

impl NeatBird {
    pub fn new(game_size: Vector, rules: GameRules, player_keys: &[GenomeKey], seed: &str) -> NeatBird {
        let seed = {
            let mut seed_array = [0u8; 32];
            seed.as_bytes().iter()
                .enumerate()
                .filter(|(index, _)| *index < 32)
                .for_each(|(index, byte)|{
                    seed_array[index] = *byte;
                });
            seed_array
        };

        let mut rng = SeedableRng::from_seed(seed);
        let mut players = Vec::new();
        for player in player_keys.iter(){
            players.push(
                (
                    Player {
                        flap_power: 0f32,
                        pos_y: 15f32,
                        width: 24f32,
                        color: Color::RED,
                        alive: true,
                    }, Evaluation{ genome: player.clone(), fitness: 0 })
            );
        };
        let covered_distance = 1;
        let walls = NeatBird::create_initial_walls(game_size, &rules, &mut rng);
        let next_wall_spacing = generate_wall_spacing(&rules, &mut rng);
        let wall_spacing_counter = 0;
        NeatBird {players, walls, game_size, covered_distance, rules, next_wall_spacing, wall_spacing_counter, rng, seed}
    }

    fn create_initial_walls(game_size: Vector, rules: &GameRules, rng: &mut impl Rng) -> VecDeque<Wall> {
        let mut walls = VecDeque::new();
        let mut next_wall_pos = rules.wall_spacing_max as f32;
        while next_wall_pos < game_size.x {
            let top_size = rng.gen_range(rules.wall_hole_pos_top_min, rules.wall_hole_pos_top_max) as f32;
            let bot_size = game_size.y as f32 - (top_size + rng.gen_range(rules.wall_hole_size_min, rules.wall_hole_size_max) as f32);
            walls.push_back(
                Wall {
                    top: top_size,
                    bot: bot_size,
                    pos_x: next_wall_pos,
                    width: rules.wall_width_min as f32,
                    color: Color::BLACK
                }
            );
            next_wall_pos += rules.wall_spacing_max as f32;
        }
        walls
    }

    pub fn update_frame(&mut self, flaps: &[bool]) -> bool {
        self.wall_spacing_counter += 1;
        self.covered_distance += 1;
        if self.wall_spacing_counter % self.next_wall_spacing == 0 {
            let wall = self.generate_wall();
            self.walls.push_back(wall);
            self.wall_spacing_counter = 0;
            self.next_wall_spacing = generate_wall_spacing(&self.rules, &mut self.rng)
        }

        let mut collisions = Vec::new();
        for i in 0 .. self.players.len(){
            let pos_y = self.players[i].0.pos_y;
            let width = self.players[i].0.width;
            collisions.push(self.check_collision(pos_y, width))
        }

        if self.players.iter().filter(|player| player.0.alive).count() <= 0 {
            return true
        }

        for (index, player) in self.players.iter_mut().enumerate() {
            if collisions[index]{
                player.0.alive = false;
                player.1.fitness = self.covered_distance;
            }
            if player.0.alive {
                if flaps.len() > index && flaps[index] {
                    player.0.flap_power += 9f32;
                }

                let new_pos_y = f32::min_max(
                    player.0.pos_y + (self.rules.gravity - player.0.flap_power),
                    0f32,
                    self.game_size.y - player.0.width
                );

                let flap_power = player.0.flap_power - self.rules.flap_decay;
                player.0.flap_power = f32::min_max(flap_power, 0f32, 9f32);
                player.0.pos_y = new_pos_y;
            }
        }
        false
    }

    pub fn generate_wall(&mut self) -> Wall {
        //TODO: Random Hole
        let wall_width = self.rng.gen_range(self.rules.wall_width_min, self.rules.wall_width_max) as f32;
        let top_size = self.rng.gen_range(self.rules.wall_hole_pos_top_min, self.rules.wall_hole_pos_top_max);
        let bot_size = self.game_size.y as usize - (top_size + self.rng.gen_range(self.rules.wall_hole_size_min, self.rules.wall_hole_size_max));
        Wall{
            top: top_size as f32,
            bot: bot_size as f32,
            pos_x: (self.covered_distance + self.game_size.x as usize) as f32,
            width: wall_width,
            color: Color::BLACK
        }
    }

    fn check_collision(&mut self, player_pos_y: f32, player_width: f32) -> bool {
        use quicksilver::geom::Shape;
        if let Some(next_column) = self.walls.front() {
            if next_column.pos_x + next_column.width < self.covered_distance as f32 {
                let _ = self.walls.pop_front();
                self.check_collision(player_pos_y, player_width)
            } else {
                let player_hitbox = Rectangle::new((self.covered_distance as f32, player_pos_y), (player_width, player_width));
                let wall_top_hitbox = Rectangle::new((next_column.pos_x, 0f32), (next_column.width, next_column.top));
                let wall_bottom_hitbox = Rectangle::new((next_column.pos_x, self.game_size.y-next_column.bot), (next_column.width, self.game_size.y));
                if player_hitbox.overlaps(&wall_top_hitbox) || player_hitbox.overlaps(&wall_bottom_hitbox) {
                    true
                } else {
                    false
                }
            }
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        let rng: StdRng = SeedableRng::from_seed(self.seed);
        self.walls = Self::create_initial_walls(self.game_size, &self.rules, &mut self.rng);
        self.covered_distance = 0;
        for player in self.players.iter_mut(){
            player.0.flap_power = 0f32;
            player.0.pos_y = 200f32;
            player.0.width = 24f32;
            player.0.color = Color::RED;
            player.0.alive = true;
        }
    }

    pub fn next_wall(&self) -> Option<&Wall> {
        self.walls.front()
    }

    pub fn walls(&self) -> impl Iterator<Item = &Wall> + '_ {
        self.walls.as_slices().0.iter().chain(self.walls.as_slices().1)
    }

    pub fn players(&self) -> &[(Player, Evaluation)] {
        &self.players
    }

    pub fn set_players(&mut self, player_keys: &[GenomeKey]) {
        self.players.clear();
        for player in player_keys.iter(){
            self.players.push(
                (
                    Player {
                        flap_power: 0f32,
                        pos_y: 15f32,
                        width: 24f32,
                        color: Color::RED,
                        alive: true,
                    }, Evaluation{ genome: player.clone(), fitness: 0 })
            );
        };
    }

    pub fn covered_distance(&self) -> usize {
        self.covered_distance
    }

    pub fn game_size(&self) -> &Vector {
        &self.game_size
    }
}

pub struct GameRules {
    pub gravity: f32,
    pub flap_power: f32,
    pub flap_decay: f32,
    pub flap_delay: usize,
    pub max_bird_flap_power: f32,
    pub wall_spacing_min: usize,
    pub wall_spacing_max: usize,
    pub wall_width_min: usize,
    pub wall_width_max: usize,
    pub wall_hole_size_min: usize,
    pub wall_hole_size_max: usize,
    pub wall_hole_pos_top_min: usize,
    pub wall_hole_pos_top_max: usize,
}

fn generate_wall_spacing(game_rules: &GameRules, rng: &mut impl Rng) -> usize {
    rng.gen_range(game_rules.wall_spacing_min, game_rules.wall_spacing_max)
}