from game import *
import time
from environment import Environment


class MinecraftEnvironment(Environment):

    window = None

    def __init__(self, evaluate_mode):
        """
        Initialize the game state
        """
        print ("Initialing in evaluate mode: ", evaluate_mode)
    
        if (evaluate_mode):
            self.window = Window(width=game_config.TEST_WINDOW_SIZE, height=game_config.TEST_WINDOW_SIZE, caption='Minecraft', resizable=False, vsync=False)
        else:
            self.window = Window(width=game_config.TRAIN_WINDOW_SIZE, height=game_config.TRAIN_WINDOW_SIZE, caption='Minecraft', resizable=False, vsync=False)
    
        self.window.set_phase(evaluate_mode)
    
        p = Player()
        self.window.set_player(p)
        p.setGame(self.window)
        world_file = "/test%d.txt" % random.randrange(10)
        p.task.generateGameWorld(world_file)
        self.window.model.loadMap(world_file)
        opengl_setup()
        return "Successfully initialized"


    def getActionSet(self):
        """
        Get a list of all the legal actions
        """
        #return LEGAL_ACTIONS
        return self.window.player.task.actions


    def getScreen(self):
        """
        Do one step of the game state and get the current screen
        """
        screen = self.window.get_screen()
        #print ("py screen")
        #print screen
        #print len(list(screen))
        #print list(screen)
        return list(screen)


    def performAction(self, action):
        """
        Perform the desired action
        """
        #print ("python act 1")
        # first apply the action
        self.window.player.performAction(action)
        #print ("python act 2")
        self.update()
        #print ("python act 3")
        # now determine the reward from it
        return float(self.window.player.getReward(action))

    
    def update(self):
        """
        Updates the game given the currently set params
        Called from act.
        """
        dt = pyglet.clock.tick()
        self.window.update(dt * 1000)
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event('on_draw')
        self.window.flip()


    def isEpisodeOver(self):
        """
        Determine if the game is over
        """
        return self.window.game_over or self.window.player.endGameEarly()


    def reset(self):
        """
        Reset the game state
        """
        self.window.reset()

