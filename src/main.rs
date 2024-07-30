use std::fs::read_dir;
use std::sync::Arc;
use std::time::Instant;
use clap::Parser;
use image::{ImageBuffer, Luma};
use imageproc::gradients::sobel_gradients;
use rayon::prelude::*;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args{
    #[arg(short, long, default_value = "images")]
    input_folder: String,
    #[arg(short, long, default_value = "images-out")]
    output_folder: String,
    #[arg(short, long)]
    // How many frames to combine into new frame (i.e. how many frames back to go)
    frames: usize,
    #[arg(short, long)]
    // How much space between each of the frames
    frame_spacing: usize,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let mut state = State::new(String::from(args.input_folder), args.frames, args.frame_spacing);

    state.process_images()
}

struct State {
    width: u32,
    height: u32,
    buf_size: usize,

    frame_count: usize,

    frames: Vec<Vec<u8>>,
    frame_pointers: Arc<Vec<usize>>,

    frames_to_combine: usize,
    frame_spacing: usize,
}

impl State {
    pub fn new(in_dir: String, frames_to_combine: usize, frame_spacing: usize) -> Self {
        let files = read_dir(in_dir).expect("Couldn't read images directory");


        let mut images: Vec<_> = files.into_iter().filter_map(|v| {
            let entry = v.unwrap();
            let path = entry.path();
            if path.is_file() {
                let image = image::open(path).unwrap();
                return Some(image.into_luma8());
            }
            None
        }).collect();
        println!("Loaded images");

        let (width, height) = images[0].dimensions();
        let frame_count = images.len();

        let frames = images.par_drain(..).map(|img| {
            let gradient = sobel_gradients(&img);

            let buf = gradient.into_raw();
            let buf: Vec<_> = buf.par_iter().map(|v| {
                *v as u8
            }).collect();
             // Box::pin(buf);
            buf
        }).collect::<Vec<_>>();
        println!("Got sobel frames");

        let frame_pointers: Arc<Vec<_>> = Arc::new(frames.iter().map(|img| {
            img.as_ptr() as usize
        }).collect());


        let buf_size = frames[0].len();

        Self {
            width,
            height,
            buf_size,
            frame_count,

            frames,
            frame_pointers,

            frames_to_combine,
            frame_spacing,
        }
    }

    pub fn process_images(&mut self) {
        println!("Started processing");
        (0..self.frame_count).into_iter().for_each(move |i| {
            unsafe {
                let constituent_frames: Vec<_> = (0..self.frames_to_combine)
                    .filter_map(|j| i.checked_sub(j * self.frame_spacing))
                    .map(|i| self.frames.get_unchecked(i))
                    .collect();

                let combine_timer = Instant::now();
                let image = combine_images(constituent_frames.as_slice(), self.buf_size, (self.width, self.height));
                println!("Combining frames for image {} took {:?}", i + 1, combine_timer.elapsed());

                image.save(format!("images-out/{:04}.jpg", i + 1)).unwrap();
            }
        });
    }
}

pub fn combine_images(images: &[&Vec<u8>], buf_size: usize, (width, height): (u32, u32)) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let mut result = vec![0u8; buf_size];
    let result_ptr = result.as_mut_ptr() as usize;

    images.par_iter().for_each(|&buf| {
        let result_ptr = result_ptr as *mut u8;

        for (i, v) in buf.iter().enumerate() {
            unsafe {
                *result_ptr.add(i) |= *v;
            }
        }
    });

    let img_buf = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, result).unwrap();
    return img_buf;
}