
def get_RDK(coh,seed,nFrames, pulse_coh=[], pulse_t=[], pulse_dur=[]):
    import cv2
    import numpy as np
    from RandomMotion import DotMotionStim
    # Loading pygame dependencies
    import pygame, tkinter, time
    import pygame.locals

    pygame.init()
    pygame.mixer.init()
    clock = pygame.time.Clock()

    # clock = pyglet.clock.Clock()
    # clock.set_fps_limit(60)

    # global window_size, screen, rewardTone, incorrectTone
    root = tkinter.Tk()
    window_size = (root.winfo_screenwidth(), root.winfo_screenheight())
    flags = pygame.DOUBLEBUF  # Using buffer screen to increase speed
    # screen = pygame.display.set_mode(window_size, vsync=1, display=1)  # Projecting on secondary screen
    screen = pygame.display.set_mode(window_size, flags=flags, display=1)  # Projecting on secondary screen      # Changed this line on March 30 2021 to see if flickering stops with no vsync


    msg = ['Nil']
    prev_msg = ['Nil']
    Dots = DotMotionStim();
    Dots.vel = 5
    Dots.radius = 13

    screen.fill((0, 0, 0))  # Display Black Screen
    pygame.display.update()
    pygame.event.pump()

    counter = 0
    FrameRate = 150
    loop_timer = time.time_ns()

    Dots.newStimulus(coh,seed)
    RDK = np.zeros((nFrames,window_size[0],window_size[1]))

    if pulse_coh: pulse=True;
    else: pulse=False

    for i in range(nFrames):

        if pulse and i == pulse_t:
            Dots.updateBurst(pulse_coh)
        if pulse and i == pulse_t + pulse_dur:
            Dots.updateBurst(coh)


        screen.fill((0, 0, 0))
        Dots.moveDots();  # Drawing dots
        for dot in range(Dots.nDots):
            pygame.draw.circle(screen, Dots.color, (int(Dots.x[dot]), int(Dots.y[dot])), Dots.radius)

        # pygame.display.update()
        RDK[i,:,:] = pygame.surfarray.array2d(screen)
        # pygame.event.pump()


    pygame.quit()
    return RDK



# if __name__ == '__main__':
#     get_RDK(coh=0, seed=1, nFrames=360, pulse_coh=-100, pulse_t=100, pulse_dur=100)