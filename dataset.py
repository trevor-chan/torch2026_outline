import torch
from torch.utils.data import Dataset, IterableDataset
import colorsys
import math
import os
import random
from typing import Optional, Tuple, Union

from image_utils import flatten_image, normalize_image, one_hot_encode
from torchvision import datasets, transforms
import tqdm


class BouncingBallVideoDataset(Dataset):
    """
    Synthetic dataset of ordered video frames from a bouncing colored ball.

    The generated frames are stored in memory as ``self.samples`` with shape
    ``[num_samples, 3, height, width]`` and values in ``[0, 1]``. By default the
    dataset returns only image tensors. Set ``return_conditioning=True`` to
    return ``(image, empty_conditioning)`` pairs for the repo's current trainer.
    """
    def __init__(
        self,
        num_samples: int = 10_000,
        image_size: Union[int, Tuple[int, int]] = 32,
        seed: int = 0,
        frame_dt: float = 0.5,
        average_bounce_time: float = 1.0,
        ball_radius: float = 2.5,
        trail_seconds: float = 2.0,
        trail_samples_per_frame: int = 10,
        bounce_jitter_degrees: float = 15.0,
        color_walk_std: float = 0.075,
        background_color: Tuple[float, float, float] = (0.02, 0.02, 0.025),
        normalize: bool = False,
        return_conditioning: bool = False,
        video_path: Optional[str] = "outputs/bouncing_ball_dataset.mp4",
        write_video: bool = True,
        raise_on_video_error: bool = False,
        video_fps: Optional[float] = None,
        video_frame_stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_samples = num_samples
        self.height, self.width = self._parse_image_size(image_size)
        self.seed = seed
        self.frame_dt = frame_dt
        self.average_bounce_time = average_bounce_time
        self.ball_radius = ball_radius
        self.trail_seconds = trail_seconds
        self.trail_samples_per_frame = max(1, int(trail_samples_per_frame))
        self.bounce_jitter_degrees = bounce_jitter_degrees
        self.color_walk_std = color_walk_std
        self.background_color = background_color
        self.normalize = normalize
        self.return_conditioning = return_conditioning
        self.video_path = video_path
        self.write_video = write_video
        self.raise_on_video_error = raise_on_video_error
        self.video_fps = video_fps if video_fps is not None else 1.0 / frame_dt
        self.video_frame_stride = max(1, video_frame_stride)
        self.dtype = dtype

        self.samples = self._generate_samples()
        self.conditioning = torch.empty(num_samples, 0, dtype=dtype)
        self.video_error = None

        if self.write_video and self.video_path is not None:
            try:
                self.save_video(self.video_path, stride=self.video_frame_stride)
            except Exception as error:
                self.video_error = error
                if self.raise_on_video_error:
                    raise
                print(f"Could not write dataset video to {self.video_path}: {error}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.normalize:
            sample = sample * 2.0 - 1.0

        if self.return_conditioning:
            return sample, self.conditioning[idx]
        return sample

    @staticmethod
    def _parse_image_size(image_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(image_size, int):
            return image_size, image_size

        if len(image_size) != 2:
            raise ValueError("image_size must be an int or a (height, width) tuple")

        height, width = image_size
        return int(height), int(width)

    def _generate_samples(self) -> torch.Tensor:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("image_size dimensions must be positive")
        if self.ball_radius <= 0:
            raise ValueError("ball_radius must be positive")
        if self.frame_dt <= 0:
            raise ValueError("frame_dt must be positive")
        if self.average_bounce_time <= 0:
            raise ValueError("average_bounce_time must be positive")

        min_x, max_x = self.ball_radius, self.width - self.ball_radius
        min_y, max_y = self.ball_radius, self.height - self.ball_radius
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("ball_radius is too large for the requested image_size")

        rng = random.Random(self.seed)
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)

        angle = rng.uniform(0.0, 2.0 * math.pi)
        direction = (math.cos(angle), math.sin(angle))
        speed = self._initial_speed(direction, max_x - min_x, max_y - min_y)
        velocity = (speed * direction[0], speed * direction[1])

        hue = rng.random()
        trail_point_count = max(
            1,
            int(round(self.trail_seconds / self.frame_dt * self.trail_samples_per_frame)),
        )
        history = []
        frames = torch.empty(
            self.num_samples,
            3,
            self.height,
            self.width,
            dtype=self.dtype,
        )

        grid_y_values = torch.arange(self.height, dtype=self.dtype)
        grid_x_values = torch.arange(self.width, dtype=self.dtype)
        try:
            grid_y, grid_x = torch.meshgrid(grid_y_values, grid_x_values, indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(grid_y_values, grid_x_values)

        for frame_idx in tqdm.tqdm(range(self.num_samples)):
            color = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            frames[frame_idx] = self._render_frame(
                x=x,
                y=y,
                color=color,
                history=history[-trail_point_count:],
                grid_x=grid_x,
                grid_y=grid_y,
            )

            next_hue = (hue + rng.gauss(0.0, self.color_walk_std)) % 1.0
            x, y, velocity, path_points = self._advance_position(
                x=x,
                y=y,
                velocity=velocity,
                rng=rng,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                start_hue=hue,
                end_hue=next_hue,
            )
            history.extend(path_points)
            if len(history) > trail_point_count:
                history = history[-trail_point_count:]
            hue = next_hue

        return frames

    def _initial_speed(self, direction, playfield_width: float, playfield_height: float) -> float:
        dominant_component = max(abs(direction[0]), abs(direction[1]), 1e-6)
        dominant_range = playfield_width if abs(direction[0]) >= abs(direction[1]) else playfield_height
        return dominant_range / (self.average_bounce_time * dominant_component)

    def _render_frame(self, x, y, color, history, grid_x, grid_y) -> torch.Tensor:
        background = torch.tensor(self.background_color, dtype=self.dtype).view(3, 1, 1)
        frame = background.expand(3, self.height, self.width).clone()

        if len(history) == 1:
            trail_x, trail_y, trail_color = history[0]
            frame = self._draw_disc(
                frame=frame,
                x=trail_x,
                y=trail_y,
                radius=self.ball_radius * 0.35,
                color=trail_color,
                alpha=0.15,
                grid_x=grid_x,
                grid_y=grid_y,
            )

        for trail_idx in range(len(history) - 1):
            x0, y0, _ = history[trail_idx]
            x1, y1, trail_color = history[trail_idx + 1]
            age = (trail_idx + 1) / len(history)
            alpha = 0.42 * (age ** 1.7)
            radius = self.ball_radius * (0.25 + 0.45 * age)
            frame = self._draw_capsule(
                frame=frame,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                radius=radius,
                color=trail_color,
                alpha=alpha,
                grid_x=grid_x,
                grid_y=grid_y,
            )

        frame = self._draw_disc(
            frame=frame,
            x=x,
            y=y,
            radius=self.ball_radius,
            color=color,
            alpha=1.0,
            grid_x=grid_x,
            grid_y=grid_y,
        )
        return frame.clamp(0.0, 1.0)

    def _draw_disc(self, frame, x, y, radius, color, alpha, grid_x, grid_y):
        distance = torch.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
        mask = ((radius + 1.0 - distance) / 1.0).clamp(0.0, 1.0) * alpha
        color_tensor = torch.tensor(color, dtype=self.dtype).view(3, 1, 1)
        return frame * (1.0 - mask) + color_tensor * mask

    def _draw_capsule(self, frame, x0, y0, x1, y1, radius, color, alpha, grid_x, grid_y):
        dx = x1 - x0
        dy = y1 - y0
        segment_length_sq = dx * dx + dy * dy

        if segment_length_sq <= 1e-9:
            return self._draw_disc(frame, x0, y0, radius, color, alpha, grid_x, grid_y)

        t = ((grid_x - x0) * dx + (grid_y - y0) * dy) / segment_length_sq
        t = t.clamp(0.0, 1.0)
        closest_x = x0 + t * dx
        closest_y = y0 + t * dy
        distance = torch.sqrt((grid_x - closest_x) ** 2 + (grid_y - closest_y) ** 2)
        mask = ((radius + 1.0 - distance) / 1.0).clamp(0.0, 1.0) * alpha
        color_tensor = torch.tensor(color, dtype=self.dtype).view(3, 1, 1)
        return frame * (1.0 - mask) + color_tensor * mask

    def _advance_position(
        self,
        x,
        y,
        velocity,
        rng,
        min_x,
        max_x,
        min_y,
        max_y,
        start_hue,
        end_hue,
    ):
        vx, vy = velocity
        path_points = []
        step_dt = self.frame_dt / self.trail_samples_per_frame

        for step_idx in range(self.trail_samples_per_frame):
            x, vx, bounced_x = self._reflect_axis(x + vx * step_dt, vx, min_x, max_x)
            y, vy, bounced_y = self._reflect_axis(y + vy * step_dt, vy, min_y, max_y)

            if bounced_x or bounced_y:
                jitter = math.radians(rng.uniform(-self.bounce_jitter_degrees, self.bounce_jitter_degrees))
                cos_jitter = math.cos(jitter)
                sin_jitter = math.sin(jitter)
                vx, vy = (
                    vx * cos_jitter - vy * sin_jitter,
                    vx * sin_jitter + vy * cos_jitter,
                )
                vx, vy = self._keep_velocity_inside_bounds(x, y, vx, vy, min_x, max_x, min_y, max_y)

            amount = (step_idx + 1) / self.trail_samples_per_frame
            hue = self._interpolate_hue(start_hue, end_hue, amount)
            path_points.append((x, y, colorsys.hsv_to_rgb(hue, 0.85, 1.0)))

        return x, y, (vx, vy), path_points

    @staticmethod
    def _interpolate_hue(start_hue, end_hue, amount):
        delta = (end_hue - start_hue + 0.5) % 1.0 - 0.5
        return (start_hue + delta * amount) % 1.0

    @staticmethod
    def _reflect_axis(position, velocity, lower, upper):
        if lower <= position <= upper:
            return position, velocity, False

        span = upper - lower
        period = 2.0 * span
        shifted = (position - lower) % period

        if shifted <= span:
            return lower + shifted, velocity, True
        return upper - (shifted - span), -velocity, True

    @staticmethod
    def _keep_velocity_inside_bounds(x, y, vx, vy, min_x, max_x, min_y, max_y):
        eps = 1e-6
        if x <= min_x + eps and vx < 0.0:
            vx = -vx
        elif x >= max_x - eps and vx > 0.0:
            vx = -vx

        if y <= min_y + eps and vy < 0.0:
            vy = -vy
        elif y >= max_y - eps and vy > 0.0:
            vy = -vy

        return vx, vy

    def save_video(self, path: str, stride: int = 1, fps: Optional[float] = None):
        """
        Save the ordered dataset frames to an MP4 file using imageio_ffmpeg.
        """
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError(
                "imageio_ffmpeg is required to write MP4 videos. "
                "Install it with `pip install imageio-ffmpeg`."
            ) from error

        stride = max(1, stride)
        fps = fps if fps is not None else self.video_fps
        frames = self._video_frames_uint8(self.samples[::stride])

        output_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(output_dir, exist_ok=True)

        writer = None
        try:
            writer = imageio_ffmpeg.write_frames(
                path,
                size=(self.width, self.height),
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                output_params=["-movflags", "+faststart"],
            )
            writer.send(None)

            for frame in frames:
                writer.send(frame.numpy().tobytes())
        except Exception as error:
            raise RuntimeError(f"imageio_ffmpeg failed while writing {path}: {error}") from error
        finally:
            if writer is not None:
                writer.close()

    @staticmethod
    def _video_frames_uint8(samples: torch.Tensor) -> torch.Tensor:
        frames = samples.detach().cpu()
        if frames.min() < 0.0:
            frames = (frames + 1.0) / 2.0

        frames = frames.clamp(0.0, 1.0)
        frames = (frames.permute(0, 2, 3, 1) * 255.0).round().to(torch.uint8)
        return frames.contiguous()

class MNISTDataset(IterableDataset):
    """
    A simplified MNIST dataset that yields samples infinitely.
    Provides raw data without transforms, using a random seed to control shuffling.
    """
    def __init__(self, data_dir='./data', train=True, seed=0):
        self.data_dir = data_dir
        self.train = train
        self.seed = seed
        
        # Download and load the training data
        if self.train:
            self.dataset = datasets.MNIST(
                root=self.data_dir, train=self.train, download=True, transform=transforms.ToTensor())
        else:
            self.dataset = datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transforms.ToTensor())
        
        # Set up randomization
        self.indices = list(range(len(self.dataset)))
        random.seed(self.seed)
        
        print(f"Dataset initialized with {'training' if train else 'test'} data, {len(self.dataset)} samples")
        
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """
        Create an iterator that yields samples one by one indefinitely.
        Uses the random seed for reproducibility in shuffling.
        """
        # Create a copy of indices and shuffle them
        indices = self.indices.copy()
        random.shuffle(indices)
        
        position = 0
        
        while True:
            # Reshuffle when we've gone through all samples
            if position >= len(indices):
                random.shuffle(indices)
                position = 0
            
            # Get the current sample
            idx = indices[position]
            image, label = self.dataset[idx]
            
            # Convert image to tensor if it's not already
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image)
            
            # Normalize and flatten the image using utility functions
            image = normalize_image(image)
            image = flatten_image(image)
            
            label = one_hot_encode(label)
            
            # Move to next position
            position += 1
            
            yield image, label 
