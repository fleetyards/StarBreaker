//! Print every clip of a .dba: index, name, controllers and frame span.
//!
//!     cargo run -p starbreaker-3d --example dba_dump -- <file.dba> [name filter]
//!
//! For checking which animation block a DBA metadata name lands on: a name
//! paired with the wrong block shows up as a frame span or bone set that does
//! not fit the motion the name describes.
use std::collections::HashSet;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: dba_dump <file.dba> [filter]");
    let filter = args.next().map(|f| f.to_lowercase());
    let bytes = std::fs::read(&path).expect("read dba");
    let db = starbreaker_3d::animation::parse_dba(&bytes).expect("parse dba");
    println!("{} clips in {path}", db.clips.len());
    for (i, clip) in db.clips.iter().enumerate() {
        if let Some(f) = &filter {
            if !clip.name.to_lowercase().contains(f) {
                continue;
            }
        }
        let bones: HashSet<u32> = clip.channels.iter().map(|c| c.bone_hash).collect();
        let last = clip
            .channels
            .iter()
            .flat_map(|c| c.rotations.iter().map(|k| k.time).chain(c.positions.iter().map(|k| k.time)))
            .fold(0.0f32, f32::max);
        let moving_pos = clip.channels.iter().filter(|c| c.positions.len() > 1).count();
        let moving_rot = clip.channels.iter().filter(|c| c.rotations.len() > 1).count();
        let mut hashes: Vec<_> = bones.iter().copied().collect();
        hashes.sort();
        println!(
            "{i:3} {:<45} channels={:3} last_frame={:5.1} moving_rot={:3} moving_pos={:3} start_pos={:?} first_bones={:08X?}",
            clip.name,
            clip.channels.len(),
            last,
            moving_rot,
            moving_pos,
            clip.start_position,
            &hashes[..hashes.len().min(3)],
        );
    }
}
