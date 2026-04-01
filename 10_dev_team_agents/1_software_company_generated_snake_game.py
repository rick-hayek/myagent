```python
import curses
import random
import time

class SnakeGame:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.height = 20
        self.width = 40  # 宽度设为40以在终端视觉上更接近正方形
        self.score = 0
        self.game_over = False
        
        # 初始化颜色
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # 蛇
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # 食物
        curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)  # 墙壁
        
        self.reset()

    def reset(self):
        """重置游戏状态"""
        self.score = 0
        self.game_over = False
        # 初始蛇位置：头部在中心，长度为3
        self.snake = [
            (self.height // 2, self.width // 2),
            (self.height // 2, self.width // 2 - 1),
            (self.height // 2, self.width // 2 - 2)
        ]
        self.direction = curses.KEY_RIGHT
        self.last_tail = None
        self.food = self._generate_food()
        
        self.stdscr.clear()
        self._draw_border()
        self._render_full()

    def _draw_border(self):
        """绘制墙壁"""
        for y in range(self.height):
            for x in range(self.width):
                if y == 0 or y == self.height - 1 or x == 0 or x == self.width - 1:
                    self.stdscr.addch(y, x, '#', curses.color_pair(3))

    def _generate_food(self):
        """在空位生成食物"""
        while True:
            food = (random.randint(1, self.height - 2), random.randint(1, self.width - 2))
            if food not in self.snake:
                return food

    def _render_full(self):
        """全量渲染（仅在初始化或重置时调用）"""
        # 画食物
        self.stdscr.addch(self.food[0], self.food[1], '*', curses.color_pair(2))
        # 画蛇
        for i, (y, x) in enumerate(self.snake):
            char = 'O' if i == 0 else 'o'
            self.stdscr.addch(y, x, char, curses.color_pair(1))
        self._update_score_display()
        self.stdscr.refresh()

    def _update_score_display(self):
        """实时更新分数显示"""
        self.stdscr.addstr(self.height, 0, f"Current Score: {self.score}")

    def update(self, key):
        """核心逻辑：更新坐标 -> 碰撞检测 -> 渲染"""
        # 1. 处理输入方向 (防止180度掉头)
        if key in [curses.KEY_UP, ord('w'), ord('W')] and self.direction != curses.KEY_DOWN:
            self.direction = curses.KEY_UP
        elif key in [curses.KEY_DOWN, ord('s'), ord('S')] and self.direction != curses.KEY_UP:
            self.direction = curses.KEY_DOWN
        elif key in [curses.KEY_LEFT, ord('a'), ord('A')] and self.direction != curses.KEY_RIGHT:
            self.direction = curses.KEY_LEFT
        elif key in [curses.KEY_RIGHT, ord('d'), ord('D')] and self.direction != curses.KEY_LEFT:
            self.direction = curses.KEY_RIGHT

        # 2. 计算新蛇头
        head_y, head_x = self.snake[0]
        if self.direction == curses.KEY_UP: head_y -= 1
        elif self.direction == curses.KEY_DOWN: head_y += 1
        elif self.direction == curses.KEY_LEFT: head_x -= 1
        elif self.direction == curses.KEY_RIGHT: head_x += 1
        
        new_head = (head_y, head_x)

        # 3. 碰撞检测 (严格执行 PRD: 撞墙或撞到自身任何一段)
        if (new_head[0] <= 0 or new_head[0] >= self.height - 1 or 
            new_head[1] <= 0 or new_head[1] >= self.width - 1 or
            new_head in self.snake):
            self.game_over = True
            # 即使死亡，也要渲染最后一帧：将蛇头画在撞击点上（如果是自残）
            # 或者保持当前画面，直接进入结算
            return

        # 4. 移动逻辑
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 10
            self.food = self._generate_food()
            self.last_tail = None # 吃到食物不删尾部
            self.stdscr.addch(self.food[0], self.food[1], '*', curses.color_pair(2))
        else:
            self.last_tail = self.snake.pop()

        # 5. 增量渲染 (消除闪烁)
        if self.last_tail:
            self.stdscr.addch(self.last_tail[0], self.last_tail[1], ' ')
        
        # 重新绘制蛇段（头部变O，旧头变o）
        self.stdscr.addch(self.snake[1][0], self.snake[1][1], 'o', curses.color_pair(1))
        self.stdscr.addch(self.snake[0][0], self.snake[0][1], 'O', curses.color_pair(1))
        
        self._update_score_display()
        self.stdscr.refresh()

    def show_game_over(self):
        """结算界面"""
        msg = " GAME OVER "
        score_msg = f" Final Score: {self.score} "
        hint_msg = " Press 'R' to Restart or 'Q' to Quit "
        
        mid_y, mid_x = self.height // 2, self.width // 2
        
        self.stdscr.attron(curses.A_REVERSE | curses.A_BOLD)
        self.stdscr.addstr(mid_y - 1, mid_x - len(msg)//2, msg)
        self.stdscr.addstr(mid_y, mid_x - len(score_msg)//2, score_msg)
        self.stdscr.addstr(mid_y + 1, mid_x - len(hint_msg)//2, hint_msg)
        self.stdscr.attroff(curses.A_REVERSE | curses.A_BOLD)
        self.stdscr.refresh()

def main(stdscr):
    # 基础配置
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(150) # 刷新频率控制在 150ms
    
    game = SnakeGame(stdscr)
    
    while True:
        if not game.game_over:
            key = stdscr.getch()
            game.update(key)
        else:
            game.show_game_over()
            # 阻塞式等待重启或退出指令
            stdscr.nodelay(False)
            while True:
                choice = stdscr.getch()
                if choice in [ord('r'), ord('R')]:
                    stdscr.nodelay(True)
                    game.reset()
                    break
                elif choice in [ord('q'), ord('Q')]:
                    return
        
if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
```