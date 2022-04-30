import cv2
import numpy
import pygame
import typing
import random


class Params:

    def __init__(self, radius):
        self.population_size = 200

        self.blur_strategy = ('gaussian', (radius, radius))
        self.fitness_calc_strategy = 'average_rgb_distance_euclidian'

        self.merge_selection_strategy = '5x_triangle'
        self.merge_strategy = 'weighted_choice_per_pixel'  # 'weighted_linear_interpolation_per_pixel'

        # chance of replacing a low-fitness image with a fresh one.
        self.fresh_gen_strategy = 'pure_random'
        self.fresh_gen_rate = 0.1

        # chance of mutating an image instead of merging it.
        self.mutation_rate = 0.00
        self.mutation_strategy = None

    def gen_fresh_image(self, target_image):
        if self.fresh_gen_strategy == 'pure_random':
            res = pygame.Surface(target_image.get_size())
            px = pygame.surfarray.pixels2d(res)
            px[:] = numpy.random.randint(0, 0xFFFFFF, px.size, dtype=px.dtype).reshape(px.shape)
            return res
        else:
            raise NotImplementedError()

    def do_blur(self, test_image):
        if self.blur_strategy[0] == 'gaussian':
            res = test_image.copy()
            px = pygame.surfarray.array3d(res)
            cv2.blur(px, ksize=self.blur_strategy[1], dst=px)
            pygame.surfarray.blit_array(res, px)
            return res
        elif self.blur_strategy[0] is None:
            return test_image.copy()
        else:
            raise NotImplementedError()

    def calc_fitness(self, blurred_test_image, target_image):
        test_px = pygame.surfarray.array2d(blurred_test_image)
        target_px = pygame.surfarray.array2d(target_image)

        if self.fitness_calc_strategy == 'average_rgb_distance_euclidian':
            max_r = numpy.maximum(target_px & 0xFF0000, test_px & 0xFF0000) // 0x010000
            min_r = numpy.minimum(target_px & 0xFF0000, test_px & 0xFF0000) // 0x010000
            max_g = numpy.maximum(target_px & 0x00FF00, test_px & 0x00FF00) // 0x000100
            min_g = numpy.minimum(target_px & 0x00FF00, test_px & 0x00FF00) // 0x000100
            max_b = numpy.maximum(target_px & 0x0000FF, test_px & 0x0000FF)
            min_b = numpy.minimum(target_px & 0x0000FF, test_px & 0x0000FF)
            dist_per_px = numpy.sqrt((max_r - min_r) ** 2 + (max_g - min_g) ** 2 + (max_b - min_b) ** 2)
            return numpy.average(dist_per_px), dist_per_px
        else:
            raise NotImplementedError()

    def make_selection(self, population, step_count, n=1):
        if self.merge_selection_strategy == '5x_triangle':
            weights = [5 - 4 * (i / len(population)) for i in range(len(population))]
            return random.choices(population, weights, k=n)
        else:
            raise NotImplementedError()

    def do_mutation(self, p):
        img, fitness, blur, dist = p
        if self.mutation_strategy is None:
            return img
        else:
            raise NotImplementedError()

    def do_merge(self, p1, p2):
        img1, fitness1, blur1, dist1 = p1
        img2, fitness2, blur2, dist2 = p2
        arr1 = pygame.surfarray.pixels2d(img1)
        arr2 = pygame.surfarray.pixels2d(img2)
        if self.merge_strategy == 'weighted_linear_interpolation_per_pixel':
            total_dist = dist1 + dist2
            total_dist[total_dist <= 0] = 1
            interp = dist1 / total_dist

            res = img1.copy()
            res_array = pygame.surfarray.pixels2d(res)
            res_array[:] = (numpy.floor((arr1 & 0xFF0000) * (1 - interp) + (arr2 & 0xFF0000) * interp) +
                            numpy.floor((arr1 & 0x00FF00) * (1 - interp) + (arr2 & 0x00FF00) * interp) +
                            numpy.floor((arr1 & 0x0000FF) * (1 - interp) + (arr2 & 0x0000FF) * interp))
            return res
        elif self.merge_strategy == "weighted_choice_per_pixel":
            total_dist = dist1 + dist2
            total_dist[total_dist <= 0] = 1
            chances = dist1 / total_dist
            rand = numpy.random.rand(chances.size).reshape(chances.shape)

            res = img1.copy()
            res_array = pygame.surfarray.pixels2d(res)
            res_array[rand > chances] = arr1[rand > chances]
            res_array[rand <= chances] = arr2[rand <= chances]
            return res
        else:
            raise NotImplementedError()

    def select_and_reproduce(self, population, step_count):
        rng = random.random()
        if rng < self.mutation_rate:
            p = self.make_selection(population, step_count)
            return self.do_mutation(p[0])
        rng -= self.mutation_rate

        if rng < self.fresh_gen_rate:
            return None
        rng -= self.fresh_gen_rate

        p1, p2 = self.make_selection(population, step_count, n=2)
        return self.do_merge(p1, p2)


class Simulation:

    def __init__(self, target_image: pygame.Surface, params: Params):
        self.target_image = target_image
        self.params = params

        # list of (image, fitness, blurred_image, per_pixel_distance)
        self.population = []
        self.step_count = 0

        self._cached_per_pix_dist_img = (-1, None)

    def step(self):
        self.step_count += 1
        if len(self.population) == 0:
            self._ensure_population_size_and_sorting()
        else:
            old_population = list(self.population)

            self.population.clear()
            for i in range(len(old_population)):
                offspring = self.params.select_and_reproduce(old_population, self.step_count)
                if offspring is not None:
                    self._add_to_population(offspring, and_sort=False)

            self._ensure_population_size_and_sorting()

    def restart(self):
        self.step_count = 0
        self.population.clear()
        self._cached_per_pix_dist_img = (-1, None)

    def get_best_image(self):
        return self.population[0][0] if len(self.population) > 0 else None

    def get_best_fitness(self):
        return self.population[0][1] if len(self.population) > 0 else -1

    def get_best(self) -> typing.Tuple[pygame.Surface, float, pygame.Surface, numpy.ndarray]:
        return self.population[0] if len(self.population) > 0 else (None, -1, None, None)

    def get_per_pixel_distance_as_image(self):
        dist = self.get_best()[3]
        if dist is None:
            self._cached_per_pix_dist_img = (-1, None)
        elif self._cached_per_pix_dist_img[0] is not self.step_count:
            bad = 255, 0x0000FF
            good = 0, 0x000000
            surf = pygame.Surface(self.target_image.get_size())
            px = pygame.surfarray.pixels2d(surf)
            px[:] = good[1] + (bad[1] - good[1]) * (dist - good[0]) / (bad[0] - good[0])
            px[dist >= bad[0]] = bad[1]  # pixels really far away are bad color
            px[dist <= good[0]] = good[1]  # pixels pretty close are good color
            self._cached_per_pix_dist_img = (self.step_count, surf)

        return self._cached_per_pix_dist_img[1]

    def _add_to_population(self, test_image, and_sort=True):
        blurred_test_image = self.params.do_blur(test_image)
        fitness, per_px_dist = self.params.calc_fitness(self.target_image, blurred_test_image)

        self.population.append((test_image, fitness, blurred_test_image, per_px_dist))

        if and_sort:
            self.population.sort(key=lambda img: img[1])

    def _ensure_population_size_and_sorting(self):
        # reduce population size (if we've gone over somehow?)
        if len(self.population) > self.params.population_size:
            self.population = self.population[:self.params.population_size]

        # increase population size by spawning fresh images
        while len(self.population) < self.params.population_size:
            new_img = self.params.gen_fresh_image(self.target_image)
            self._add_to_population(new_img, and_sort=False)

        self.population.sort(key=lambda img: img[1])


class Simulation2:

    def __init__(self, target: pygame.Surface, blur_func):
        self.target = target
        self.target_channels = [
            pygame.surfarray.array_red(self.target),
            pygame.surfarray.array_green(self.target),
            pygame.surfarray.array_blue(self.target),
        ]

        self.blur_func = blur_func

        self.img = target.copy()
        self.blurred_img = None

        self.distance = None
        self.blurred_distance = None
        self.anti_distance = None
        self.blurred_anti_distance = None

        self.fitness = None
        self.step_count = None

        self.restart()

    def get_output(self):
        return self.img

    def get_fitness(self):
        return self.fitness

    def restart(self):
        self.img = self.target.copy()
        self.blurred_img = None
        self.distance = None
        self.blurred_distance = None
        self.anti_distance = None
        self.blurred_anti_distance = None
        self.fitness = -1
        self.step_count = 0
        self._calc_derived_images()

    def step(self):
        blur_dist_array = pygame.surfarray.pixels3d(self.blurred_distance)
        blur_anti_dist_array = pygame.surfarray.pixels3d(self.blurred_anti_distance)
        ratio = max(2, 5 * (1 - self.step_count / 300))

        new_img_int8 = pygame.surfarray.array3d(self.img)
        new_img = new_img_int8.astype(numpy.float64)
        rand = numpy.random.rand(*new_img.shape)

        new_img[:] = new_img + blur_dist_array * (rand * ratio)
        new_img[:] = new_img - blur_anti_dist_array * (rand * ratio)
        new_img[:] = numpy.minimum(new_img, 255.999)
        new_img[:] = numpy.maximum(new_img, 0)

        new_img_int8[:] = new_img.astype(numpy.int8, casting='unsafe')
        pygame.surfarray.blit_array(self.img, new_img_int8)

        self._calc_derived_images()
        self.step_count += 1

    def _calc_derived_images(self):
        self.blurred_img = self.blur_func(self.img)
        self.distance, self.anti_distance = self._calc_distance_from_target(self.blurred_img)
        self.blurred_distance = self.blur_func(self.distance)
        self.blurred_anti_distance = self.blur_func(self.anti_distance)

        self.fitness = (numpy.mean(pygame.surfarray.pixels3d(self.distance))
                        + numpy.mean(pygame.surfarray.pixels3d(self.anti_distance))) / 3

    def _calc_distance_from_target(self, img):
        res = img.copy()
        for i, channel in enumerate(self._get_color_channel_refs(res)):
            too_low = self.target_channels[i] > channel
            channel[too_low] = (self.target_channels[i] - channel)[too_low]
            channel[~too_low] = 0

        res_anti = img.copy()
        for i, channel in enumerate(self._get_color_channel_refs(res_anti)):
            too_high = channel > self.target_channels[i]
            channel[too_high] = (channel - self.target_channels[i])[too_high]
            channel[~too_high] = 0

        return res, res_anti

    def _get_color_channel_refs(self, img):
        return [
            pygame.surfarray.pixels_red(img),
            pygame.surfarray.pixels_green(img),
            pygame.surfarray.pixels_blue(img)
        ]


def get_box_filter_func(radius):
    def box_filter(img: pygame.Surface) -> pygame.Surface:
        res = img.copy()
        px = pygame.surfarray.array3d(res)
        cv2.blur(px, ksize=(radius, radius), dst=px)
        pygame.surfarray.blit_array(res, px)
        return res
    return box_filter


if __name__ == "__main__":
    # s = pygame.Surface((10, 10))
    # s.fill((255, 255, 255))
    # pygame.draw.circle(s, (0, 0, 0), s.get_rect().center, 3)
    # blurred = Params(3).do_blur(s)
    # pygame.image.save(blurred, "data/3x3_circle_in_10x10.png")

    # s = pygame.image.load("data/splash.png")
    # blurred = Params(15).do_blur(s)
    # pygame.image.save(blurred, "data/splash_blurred_15.png")

    pygame.init()

    # original = pygame.Surface((10, 10))
    # original.fill((255, 255, 255))
    # pygame.draw.circle(original, (0, 0, 0), original.get_rect().center, 3)

    img_file = "data/splash_blurred_15.png"  # "data/3x3_circle_in_10x10.png"  #
    target_image = pygame.image.load(img_file)
    original = pygame.image.load("data/splash.png")

    simulation = Simulation2(target_image, get_box_filter_func(15))

    W, H = target_image.get_size()

    screen = pygame.display.set_mode((W * 4, H * 2), pygame.SCALED | pygame.RESIZABLE)
    clock = pygame.time.Clock()

    auto_step = True
    pause_at = -1
    modes = ('normal', 'blurred', 'distance', 'anti_distance', 'target', 'original')
    mode = 0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    simulation.step()
                    auto_step = False
                    print(f"INFO: Best fitness at step {simulation.step_count} is {simulation.get_fitness()}")
                elif e.key == pygame.K_r:
                    print("INFO: restarting simulation")
                    auto_step = True
                    simulation.restart()
                elif e.key == pygame.K_m:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        mode = (mode - 1) % len(modes)
                    else:
                        mode = (mode + 1) % len(modes)
                    print(f"INFO: setting viewing mode to {modes[mode]}")

        if auto_step:
            simulation.step()
            print(f"INFO: Best fitness at step {simulation.step_count} is {simulation.get_fitness()}")

            if pause_at == simulation.step_count:
                auto_step = False

        screen = pygame.display.get_surface()

        to_blit = [
            [simulation.img, original, simulation.distance, simulation.anti_distance],
            [simulation.blurred_img, target_image, simulation.blurred_distance, simulation.blurred_anti_distance]
        ]
        screen.fill((0, 0, 0))
        for y in range(len(to_blit)):
            for x in range(len(to_blit[0])):
                if to_blit[y][x] is not None:
                    screen.blit(to_blit[y][x], (x * W, y * H))

        pygame.display.flip()
        clock.tick(15)