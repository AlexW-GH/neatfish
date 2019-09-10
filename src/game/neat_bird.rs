use quicksilver::geom::{Vector, Rectangle};
use crate::game::model::{Player, Wall};
use quicksilver::graphics::Color;
use crate::game::MinMax;
use std::collections::vec_deque::VecDeque;
use rand::Rng;
use rand::rngs::ThreadRng;
use std::cell::RefCell;

pub struct NeatBird {
    walls: VecDeque<Wall>,
    players: Vec<Player>,
    game_size: Vector,
    covered_distance: usize,
    rules: GameRules,
    next_wall_spacing: usize,
    wall_spacing_counter: usize,
    rng: ThreadRng,
}

impl NeatBird {
    pub fn new(game_size: Vector, game_rules: GameRules, player_count: usize, mut rng: ThreadRng) -> NeatBird {
        let mut players = Vec::new();
        for i in 0 .. player_count{
            players.push(
                Player {
                    flap_power: 0f32,
                    pos_y: 15f32,
                    width: 24f32,
                    color: Color::RED,
                    alive: true,
                }
            );
        };
        let covered_distance = 1;
        let mut walls = VecDeque::new();
        walls.push_back(
            Wall{
                top: game_size.y/3f32,
                bot: game_size.y/3f32,
                pos_x: game_size.x/2f32,
                width: game_size.x/3f32,
                color: Color::BLACK
            }
        );
        let next_wall_spacing = generate_wall_spacing(&game_rules, &mut rng);
        let wall_spacing_counter = 0;
        NeatBird {players, walls, game_size, covered_distance, rules: game_rules, next_wall_spacing, wall_spacing_counter, rng}
    }

    pub fn update_frame(&mut self, flaps: &[bool]){
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
            let pos_y = self.players[i].pos_y;
            let width = self.players[i].width;
            collisions.push(self.check_collision(pos_y, width))
        }

        if self.players.iter().filter(|player| player.alive).count() <= 0 {
            self.reset();
        }

        for (index, player) in self.players.iter_mut().enumerate() {
            if collisions[index]{
                player.alive = false;
            }
            if player.alive {
                if flaps.len() > index && flaps[index] {
                    player.flap_power += 9f32;
                }

                let new_pos_y = f32::min_max(
                    player.pos_y + (self.rules.gravity - player.flap_power),
                    0f32,
                    self.game_size.y - player.width
                );

                let flap_power = player.flap_power - self.rules.flap_decay;
                player.flap_power = f32::min_max(flap_power, 0f32, 9f32);
                player.pos_y = new_pos_y;
            }
        }
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

    fn reset(&mut self) {
        self.walls = VecDeque::new();
        self.walls.push_back(
            Wall{
                top: self.game_size.y/3f32,
                bot: self.game_size.y/3f32,
                pos_x: self.game_size.x/1.5f32,
                width: self.game_size.x/4f32,
                color: Color::BLACK
            }
        );
        self.covered_distance = 0;
        for player in self.players.iter_mut(){
            player.flap_power = 0f32;
            player.pos_y = 200f32;
            player.width = 24f32;
            player.color = Color::RED;
            player.alive = true;
        }
    }

    pub fn walls(&self) -> impl Iterator<Item = &Wall> + '_ {
        self.walls.as_slices().0.iter().chain(self.walls.as_slices().1)
    }

    pub fn players(&self) -> &[Player] {
        &self.players
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

fn generate_wall_spacing(game_rules: &GameRules, rng: &mut ThreadRng) -> usize {
    rng.gen_range(game_rules.wall_spacing_min, game_rules.wall_spacing_max)
}