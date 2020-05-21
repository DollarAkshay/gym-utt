
import time
import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import os
import signal
import sys

try:
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: install pygame using `pip install pygame`".format(e))

SIZE = 3


class UTTEnv(gym.Env):

    metadata = {'render.modes': ['human'],
                'video.frames_per_second': 120}

    def __init__(self):

        self.action_space = spaces.Discrete(SIZE**4)

        # The first 81 cells will represent the last move made,
        # next 81 will be the player board
        # the next 81 will represent the opponent board
        self.observation_space = spaces.MultiBinary(3 * (SIZE**4))
        self.player_chars = ['X', 'O']
        self.player_colors = [(35, 160, 255), (255, 180, 20)]
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

        self.reset()

    def step(self, move):
        """
        This method steps the game forward one step and
        shoots a bubble at the given angle.
        Parameters
        ----------
        action : int
            The action is an angle between 0 and 180 degrees, that
            decides the direction of the bubble.
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (list) :
                seperate rewards for player 1 and player 2
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                diagnostic information useful for debugging.
        """
        reward = [0, 0]
        p_id = self.player_turn
        o_id = 1 if p_id == 0 else 0
        self.small_board_won = False
        valid = self.makeMove(move)

        if valid == False:
            reward[p_id] = -100
            reward[o_id] = +100
            return self._get_obs(p_id), reward, True, {"msg": "Invalid Move"}

        if self.big_board_won == True:
            reward[p_id] = +100
            reward[o_id] = -100
            return self._get_obs(p_id), reward, True, {"msg": "Won Game"}

        if self.getMovesLeft() == 0:
            reward = self.getCountRewards()
            return self._get_obs(p_id), reward, True, {"msg": "Draw Game"}

        if self.small_board_won == True:
            reward[p_id] = +5
            reward[o_id] = -5
            return self._get_obs(p_id), reward, False, {"msg": "Won Small Board"}

        return self._get_obs(p_id), reward, False, {}

    def reset(self):
        self.screen = None
        self.player_turn = 0

        self.big_board_won = False

        self.last_move = -1
        self.big_board = [[[0]*(SIZE**2) for _ in range(SIZE**2)] for __ in range(2)]
        self.small_board = [[[0]*SIZE for _ in range(SIZE)] for __ in range(2)]

        return self._get_obs(0)

    def _get_obs(self, p_id):

        obs = []

        last_move_arr = [0]*(SIZE**4)
        if self.last_move != -1:
            last_move_arr[self.last_move] = 1
        obs.extend(last_move_arr)

        if p_id == 0:
            obs.extend(np.array(self.big_board[0]).flatten())
            obs.extend(np.array(self.big_board[1]).flatten())
        else:
            obs.extend(np.array(self.big_board[1]).flatten())
            obs.extend(np.array(self.big_board[0]).flatten())

        return obs

    def makeMove(self, move):

        row = move // (SIZE**2)
        col = move % (SIZE**2)
        b_row = row//SIZE
        b_col = col//SIZE
        valid_move = False

        lm_row = self.last_move // (SIZE**2)
        lm_col = self.last_move % (SIZE**2)
        n_b_row = lm_row % SIZE
        n_b_col = lm_col % SIZE

        full_board_move = False

        # Check if player can play in the full board
        if self.last_move == -1 or \
           (self.small_board[0][n_b_row][n_b_col] == 1 or self.small_board[1][n_b_row][n_b_col] == 1) or \
           self.getSmallBoardMoves(n_b_row, n_b_col) == 0:
            full_board_move = True

        if self.big_board[0][row][col] == 0 and self.big_board[1][row][col] == 0 and \
           self.small_board[0][b_row][b_col] == 0 and self.small_board[1][b_row][b_col] == 0 and \
           (full_board_move == True or (b_row == n_b_row and b_col == n_b_col)):

            valid_move = True
            self.big_board[self.player_turn][row][col] = 1
            self.updateSmallBoard(move)

        self.last_move = move

        self.player_turn = 1 if self.player_turn == 0 else 0

        return valid_move

    def updateSmallBoard(self, move):

        row = move // (SIZE**2)
        col = move % (SIZE**2)
        b_row = row//SIZE
        b_col = col//SIZE
        p_id = self.player_turn
        board_won = False

        # Check row wise
        if board_won == False:
            for i in range(b_row*SIZE, b_row*SIZE+SIZE):
                count = 0
                for j in range(b_col*SIZE, b_col*SIZE+SIZE):
                    if self.big_board[p_id][i][j] == 1:
                        count += 1
                if count == SIZE:
                    board_won = True
                    break

        # Check column wise
        if board_won == False:
            for j in range(b_col*SIZE, b_col*SIZE+SIZE):
                count = 0
                for i in range(b_row*SIZE, b_row*SIZE+SIZE):
                    if self.big_board[p_id][i][j] == 1:
                        count += 1
                if count == SIZE:
                    board_won = True
                    break

        # Top-Left to Bottom-Right diagonal
        if board_won == False:
            count = 0
            for i in range(SIZE):
                if self.big_board[p_id][b_row*SIZE+i][b_col*SIZE+i] == 1:
                    count += 1
            if count == SIZE:
                board_won = True

        # Top-Right to Bottom-left diagonal
        if board_won == False:
            count = 0
            for i in range(SIZE):
                if self.big_board[p_id][(b_row+1)*SIZE-1-i][(b_col+1)*SIZE-1-i] == 1:
                    count += 1
            if count == SIZE:
                board_won = True

        if board_won == True:
            self.small_board[p_id][b_row][b_col] = 1
            self.small_board_won = True
            self.checkWinner()

    def checkWinner(self):

        p_id = self.player_turn

        # Check row wise
        for i in range(SIZE):
            count = 0
            for j in range(SIZE):
                if self.small_board[p_id][i][j] == 1:
                    count += 1
            if count == SIZE:
                self.big_board_won = True
                return

        # Check column wise
        for j in range(SIZE):
            count = 0
            for i in range(SIZE):
                if self.small_board[p_id][i][j] == 1:
                    count += 1
            if count == SIZE:
                self.big_board_won = True
                return

        # Check diagonals
        count = 0
        for i in range(SIZE):
            if self.small_board[p_id][i][i] == 1:
                count += 1
        if count == SIZE:
            self.big_board_won = True
            return

        count = 0
        for i in range(SIZE):
            if self.small_board[p_id][i][SIZE - i - 1] == 1:
                count += 1
        if count == SIZE:
            self.big_board_won = True
            return

    def getCountRewards(self):
        p0 = np.sum(self.small_board[0])
        p1 = np.sum(self.small_board[1])

        if p0 > p1:
            return [+100, -100]
        elif p1 > p0:
            return [-100, +100]
        else:
            return [0, 0]

    def getSmallBoardMoves(self, row, col):
        move_count = 0
        for i in range(SIZE):
            for j in range(SIZE):
                ri = row*SIZE + i
                rj = col*SIZE + j
                if self.big_board[0][ri][rj] == 0 and self.big_board[1][ri][rj] == 0:
                    move_count += 1
        return move_count

    def getMovesLeft(self):

        # lm_row = self.last_move // (SIZE**2)
        # lm_col = self.last_move % (SIZE**2)
        # n_b_row = lm_row % SIZE
        # n_b_col = lm_col % SIZE

        # self.last_move

        # # Check if player Can play in the full board
        # if self.last_move == -1 or \
        #    (self.small_board[0][n_b_row][n_b_col] == 1 or self.small_board[1][n_b_row][n_b_col] == 1) or \
        #    getSmallBoardMoves(n_b_row, n_b_col) == 0:

        move_count = 0
        for i in range(SIZE**2):
            for j in range(SIZE**2):
                bi = i//SIZE
                bj = j//SIZE
                if self.big_board[0][i][j] == 0 and self.big_board[1][i][j] == 0 and \
                   self.small_board[0][bi][bj] == 0 and self.small_board[1][bi][bj] == 0:
                    move_count += 1
        return move_count

    def getValidMoves(self):

        lm_row = self.last_move // (SIZE**2)
        lm_col = self.last_move % (SIZE**2)
        n_b_row = lm_row % SIZE
        n_b_col = lm_col % SIZE

        move_list = []

        # Check if player can play in the full board
        if self.last_move == -1 or \
           (self.small_board[0][n_b_row][n_b_col] == 1 or self.small_board[1][n_b_row][n_b_col] == 1) or \
           self.getSmallBoardMoves(n_b_row, n_b_col) == 0:

            for i in range(SIZE**2):
                for j in range(SIZE**2):
                    bi = i//SIZE
                    bj = j//SIZE
                    if self.big_board[0][i][j] == 0 and self.big_board[1][i][j] == 0 and \
                            self.small_board[0][bi][bj] == 0 and self.small_board[1][bi][bj] == 0:
                        move_list.append(i*(SIZE**2) + j)
        else:
            for i in range(n_b_row*SIZE, (n_b_row+1)*SIZE):
                for j in range(n_b_col*SIZE, (n_b_col+1)*SIZE):
                    if self.big_board[0][i][j] == 0 and self.big_board[1][i][j] == 0:
                        move_list.append(i*(SIZE**2) + j)

        return move_list

    def render(self, mode='human', close=False):

        if mode == 'console':
            print(self._get_game_state)
        elif mode == "human":

            margin = 50
            draw_width = 900
            draw_height = 900

            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    screen_width = draw_width + 2 * margin
                    screen_height = draw_height + 2 * margin
                    self.screen = pygame.display.set_mode((screen_width, screen_height))

                clock = pygame.time.Clock()
                self.screen.fill((255, 255, 255))

                # Draw Big Board
                big_gap = round(draw_width / SIZE)
                for i in range(1, SIZE):

                    # Horizontal Lines
                    pygame.gfxdraw.line(self.screen, margin, margin + big_gap * i-1, margin + draw_width, margin + big_gap * i-1, (0, 0, 0))
                    pygame.gfxdraw.line(self.screen, margin, margin + big_gap * i+0, margin + draw_width, margin + big_gap * i+0, (0, 0, 0))
                    pygame.gfxdraw.line(self.screen, margin, margin + big_gap * i+1, margin + draw_width, margin + big_gap * i+1, (0, 0, 0))

                    # Vertical Lines
                    pygame.gfxdraw.line(self.screen, margin + big_gap * i-1, margin + draw_height, margin + big_gap * i-1, margin, (0, 0, 0))
                    pygame.gfxdraw.line(self.screen, margin + big_gap * i+0, margin + draw_height, margin + big_gap * i+0, margin, (0, 0, 0))
                    pygame.gfxdraw.line(self.screen, margin + big_gap * i+1, margin + draw_height, margin + big_gap * i+1, margin, (0, 0, 0))

                # Draw Winners of Big Boards
                font = pygame.font.SysFont("NotoSans-Regular.ttf", round(big_gap))
                for k in range(2):
                    for i in range(SIZE):
                        for j in range(SIZE):
                            if self.small_board[k][i][j] == 1:
                                p_char = self.player_chars[k]
                                text = font.render(p_char, True, self.player_colors[k])
                                text_rect = text.get_rect(center=(margin + big_gap * (j+0.5), margin + big_gap * (i+0.5)))
                                self.screen.blit(text, text_rect)

                # Draw Small Boards
                small_gap = round(draw_width / (SIZE**2))
                small_margin = 10
                for i in range(SIZE):
                    for j in range(SIZE):
                        if self.small_board[0][i][j] == 0 and self.small_board[1][i][j] == 0:
                            for k in range(1, SIZE):
                                # Horizontal Lines
                                pygame.gfxdraw.line(self.screen,
                                                    margin + j*big_gap + small_margin,
                                                    margin + i*big_gap + small_gap * k,
                                                    margin + (j+1)*big_gap - small_margin,
                                                    margin + i*big_gap + small_gap * k,
                                                    (80, 80, 80))
                                # Vertical Lines
                                pygame.gfxdraw.line(self.screen,
                                                    margin + j*big_gap + small_gap * k,
                                                    margin + i*big_gap + small_margin,
                                                    margin + j*big_gap + small_gap * k,
                                                    margin + (i+1)*big_gap - small_margin,
                                                    (80, 80, 80))

                font = pygame.font.SysFont("NotoSans-Regular.ttf", round(small_gap * 0.8))
                for k in range(2):
                    for i in range(SIZE**2):
                        for j in range(SIZE**2):
                            b_row = i//SIZE
                            b_col = j//SIZE
                            if self.small_board[0][b_row][b_col] == 0 and self.small_board[1][b_row][b_col] == 0:
                                if self.big_board[k][i][j] == 1:
                                    p_char = self.player_chars[k]
                                    text = font.render(p_char, True, self.player_colors[k])
                                    text_rect = text.get_rect(center=(margin + small_gap * (j+0.5), margin + small_gap * (i+0.5)))
                                    self.screen.blit(text, text_rect)

                pygame.display.flip()
                self.signal_handler()
                clock.tick(self.metadata["video.frames_per_second"])

    def close(self):
        pygame.quit()

    def signal_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    print("pressed CTRL-C as an event")
                    self.close()
