#个人python学习作，在经过大学的人工智能学习后，我对自己大一时的这个C课设作品有了一些新的想法，遂移植到python上继续深耕


import pygame                 # 游戏开发库，负责所有图形界面和交互
import sys                    # 与Python解释器交互，关窗口的
import numpy as np            # 高效处理二维数组（地图数据），比列表好用
import json                   # 保存和读取关键点数据（让程序有"记忆"）
import os                     # 检查文件是否存在、路径操作，保证程序重启后能记住之前学到的经验
from collections import deque # BFS没学好最痛苦的一集，当复习了，先进先出还是用双端队列最合适
import random                 # 生成随机数
import heapq                  # A*规划需要每次取出代价最小的元素，故选用优先队列

# 初始化pygame
pygame.init()

# 颜色定义（RGB）
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

class KeyPoint:
    # 关键点类，存储重要的路标信息（关键点 = 路标）
    def __init__(self, x, y):
        self.x = x
        self.y = y
#贪心算法遗留
#        self.direction_changes = direction_changes  # 方向改变次数
#        self.best_directions = {}  # 是个字典，没有顺序需求，存储到不同目标的最佳初始方向，如point.best_directions = {(5, 5), (1, 0)}
#        # 每个目标点存一个最佳目标也太蠢了，我得想办法让程序记住区域，用一个点就能代表整片区域
#        # p1 = KeyPoint(10, 20)，存路标的
        self.next_landmarks = {}        # {目标区域: (下一个关键点x, 下一个关键点y)}
        self.distance_to_regions = {}   # {目标区域: 曼哈顿距离}
        
    def to_dict(self):
        return {'x': self.x, 'y': self.y, 
                'next_landmarks': self.next_landmarks,
                'distance_to_regions': self.distance_to_regions}
    # 将kp对象转成字典，因为对象没办法存入json文件中，所以需要先提取出列表self.key_points中的kp对象将其转为字典，
    # 再把这些字典存入名为data的列表中统一转成json文件
    
    @classmethod
    def from_dict(cls, data):
        kp = cls(data['x'], data['y'])
        kp.next_landmarks = data.get('next_landmarks', {})
        kp.distance_to_regions = data.get('distance_to_regions', {})
        return kp
    # 没对象只能用类方法，从json文件转出的列表提取出字典后重新转回kp对象使用，point = KeyPoint.from_dict(data)

class Car:
    # 小车类
    def __init__(self, x, y, size=5):
        self.x = x  # 中心点x坐标
        self.y = y  # 中心点y坐标
        self.size = size  # 小车大小（边长），固定了，不想改
    # car = Car(10, 20)
        
    def get_cells(self):
        # 获取小车占据的所有格子，用来检测撞墙的
        cells = []
        half = self.size // 2
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                cells.append((self.x + i, self.y + j))
        return cells

class MapNavigation:
    def __init__(self, width, height, map_name="default", cell_size=5):
        self.width = width   # 地图宽度（格子数）
        self.height = height # 地图高度（格子数）
        self.map_name = map_name    # 增加对地图的命名，以区分保存地图及其路标用的json文件
        self.cell_size = cell_size  # 每个格子像素大小（5px）
        self.window_width = width * cell_size # 窗口宽度 = 格子数 × 格子大小
        self.window_height = height * cell_size + 120  # 窗口宽度 = 格子数 × 格子大小 + 120，多出120像素用于放按钮
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("智能小车导航系统")
        
        # 地图数据 (True=可通过, False=障碍物)
        self.map_data = np.ones((height, width), dtype=bool) # 重要的数据结构，整个地图信息基本依靠这个二维数组
        
        # 小车对象
        self.car = None
        
        # 目标点
        self.target = None

        # A* 路径缓存
        self.current_path = []
        self.final_target = None
#        self.path_start = None

        # 创建存档文件夹
        self.save_dir = "saves"                    # 文件夹名称
        os.makedirs(self.save_dir, exist_ok=True)  # 创建名为saves的文件夹，若已存在，不报错

        # 文件放在 saves 文件夹里
        self.save_file = os.path.join(self.save_dir, f"key_points_{map_name}.json")  # 连接查找路径
        
        # 关键点数据库
        self.key_points = []      # 列表，遍历所有关键点
        self.key_point_dict = {}  # 字典，快速查找 (x,y) -> KeyPoint

        # 怎么会有人自己找了一堆关键点然后没有写用他们的方法的
        self.landmark_graph = {}  # {(from_x, from_y, to_x, to_y): 关键点之间的A*路径列表}
        self.region_size = 10     # 区域划分大小（与热力图网格一致）

        # 腐蚀图，用于 find_white_region_centers
        self.eroded = None  
        
        # 移动方向 (x, y)
        self.directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]  # 八方向移动
        
        # 状态变量
        self.drawing = False      # 是否正在绘制，由鼠标拖拽状态决定
        self.editing = True       # 编辑模式，True就是编辑地图模式，False就是小车移动模式
        self.moving = False       # 小车是否在移动
        self.last_move_time = 0   # 还没动，所以上次移动的时间点为0
        self.move_interval = 100  # 单位为毫秒，这两个变量负责让小车的移动不会变成飘移
        
        # 工具选择
        self.current_tool = "brush"  # "brush画笔"或"fill填充"
        
        # 路径记忆（用于学习）
#        self.path_history = []       # 存储走过的路径点
#        self.direction_history = []  # 存储方向改变的历史
        
        # 加载对应地图的关键点数据（调用）
        self.load_key_points()
        
        # 按钮区域（稍微做了下对窗口大小的自适应）
        
        button_height = 30  # 按钮高度
        row_spacing = 20    # 纵向间隔高度
        
        # 第一排按钮的Y坐标（窗口高度 - 100）
        row1_y = self.window_height - 100
        # 第二排按钮的Y坐标（窗口高度 - 50）
        row2_y = self.window_height - 50
        
        # 根据窗口宽度计算按钮和间隔的宽度
        btn_w = int(0.19 * self.window_width)   # 按钮宽度
        gap = int(0.048 * self.window_width)    # 横向间隔宽度

        # 兜底
        btn_w = max(60, min(btn_w, 120))
        gap = max(5, gap)

        # 第一排按钮（4个）的X坐标
        # 布局：间隔-按钮1-间隔-按钮2-间隔-按钮3-间隔-按钮4-间隔
        x1 = gap
        x2 = x1 + btn_w + gap
        x3 = x2 + btn_w + gap
        x4 = x3 + btn_w + gap

        # 第一排按钮
        self.button_rect = pygame.Rect(x1, row1_y, btn_w, button_height)   # 绘制完成
        self.clear_button = pygame.Rect(x2, row1_y, btn_w, button_height)  # 清除地图
        self.reset_button = pygame.Rect(x3, row1_y, btn_w, button_height)  # 重置小车
        self.train_button = pygame.Rect(x4, row1_y, btn_w, button_height)  # 离线训练
        
        # 第二排按钮
        self.brush_button = pygame.Rect(x2, row2_y, btn_w, button_height) #画笔
        self.fill_button = pygame.Rect(x3, row2_y, btn_w, button_height) #填充
    
     # 淦，又是一对一傻瓜代码，换张地图旧关键点就得全部删掉呗，哦地图还没法保存删了就是删了
     #def load_key_points(self):
        # 加载关键点数据（定义）
        #try:
        # 防止程序因错误而崩溃
            #if os.path.exists('key_points.json'):# 当我意识到这玩意指定固定文件名的时候已经气笑了
                #with open('key_points.json', 'r') as f:
                    #data = json.load(f)
                    #for kp_data in data:
                        ##kp = KeyPoint.from_dict(kp_data)
                        #self.key_points.append(kp)
                        #self.key_point_dict[(kp.x, kp.y)] = kp
                #print(f"已加载 {len(self.key_points)} 个关键点")
        #except Exception as e:
            #print(f"加载关键点失败: {e}")

    def load_key_points(self):
        # 加载关键点和地图尺寸数据（定义）
        try:
        # 防止程序因错误而崩溃
            if os.path.exists(self.save_file):                      # 根据路径从 py文件所在文件夹/saves/ 查找对应名称的json文件
                with open(self.save_file, 'r') as f:                # 将从json文件中读出的数据视为f
                    data = json.load(f)                             # 读出关键点列表
                    # 读取地图
                    if "width" in data and "height" in data:
                        self.width = data["width"]
                        self.height = data["height"]
                    if "map_data" in data:
                        self.map_data = np.array(data["map_data"], dtype=bool)
                    # 读取关键点
                    for kp_data in data.get("key_points", []):                            # kp_data是字典
                        kp = KeyPoint.from_dict(kp_data)            # 把字典转回kp对象
                        self.key_points.append(kp)                  # 把kp对象从末端加入列表
                        self.key_point_dict[(kp.x, kp.y)] = kp      # 将kp对象中的坐标信息作为元组加入字典中
                print(f"已加载 {len(self.key_points)} 个关键点")
        except Exception as e:
            print(f"加载关键点失败: {e}")                           # 输出报错原因
    
    def save_key_points(self):
        # 保存关键点和地图尺寸数据
        try:
            data = {
                "width": self.width,
                "height": self.height,
                "map_data": self.map_data.tolist(),  # 保存地图，numpy 转列表
                "key_points": [kp.to_dict() for kp in self.key_points]}  # 把kp对象转成字典存入名为data的列表中
            with open(self.save_file, 'w') as f:                         # 根据路径“open”这么一个json文件，如果没有就创建
                                                                         # w是写入模式，打开的文件当作f
                json.dump(data, f, indent=2)                             # 把python数据列表data转换成json格式并写入文件f，缩进2空格
            print(f"已保存地图和 {len(self.key_points)} 个关键点")
        except Exception as e:
            print(f"保存失败: {e}")                           # 输出报错原因

    def flood_fill(self, x, y, fill_color):
        # 泛洪填充算法，逻辑判断与显示上色分离
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        
        # 检查点击的位置
        if fill_color == BLACK:           # 如果要填充障碍物
            if not self.map_data[y][x]:   # 已经是障碍物（not False == True）
                return
            target_value = True           # 要替换的值（白色）
            new_value = False             # 新值（黑色）
        else:                             # 如果要填充可通行区域
            if self.map_data[y][x]:       # 已经是可通行(True)
                return
            target_value = False          # 要替换的值（黑色）
            new_value = True              # 新值（白色）
        
        # 使用队列进行BFS填充，广度优先，确保不漏
        queue = deque()
        queue.append((x, y))   # BFS先进先出，双端队列方便
        visited = set()        # set()是一个不容许重复元素的集合，用于记录已经访问的点，防止搜索在一个圈里无限循环
        visited.add((x, y))    # 后续循环不会再记录初始点，所以一定要在循环前就记录上，否则虽然能正常填充，
                               # 但输出的已填充的点的数量会少一个
        
        while queue:
        # 当queue内元素全部被弹出之后，停止循环
            cx, cy = queue.popleft()  # 把初始点坐标从双端队列起点弹出
            
            # 填充当前格子
            self.map_data[cy][cx] = new_value
            
            # 检查四个方向的邻居
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) not in visited and 
                    self.map_data[ny][nx] == target_value):   # 点要在画布中，不能被记录，不能还是未填充状态
                    visited.add((nx, ny))                     # 符合要求的点记录在visited中
                    queue.append((nx, ny))                    # 这个点从末端添加到双端队列中，用于搜索它的邻居
        
        print(f"填充完成，共填充了 {len(visited)} 个格子")
    
    def is_valid_position(self, x, y):
        # 检查位置是否有效（在边界内且不是障碍物）
        half = 2  # 小车半径
        if x - half < 0 or x + half >= self.width or y - half < 0 or y + half >= self.height:  # 边界检测
            return False
        
        # 检查小车占据的所有格子
        test_car = Car(x, y)
        for cx, cy in test_car.get_cells():
            if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:  # 有点重复，但留着也没啥
                return False
            if not self.map_data[cy][cx]:  # 障碍物，not False == True
                return False
        return True

    def a_star(self, start_x, start_y, target_x, target_y):
        # A*路径规划（贪心移动不足太多了，向A*投降）
        if not self.is_valid_position(target_x, target_y):
            return None  # 终点不在有效位置还想规划路径？
        
        heap = []                                                           # 初始化优先队列
        heapq.heappush(heap, (0, start_x, start_y, [(start_x, start_y)]))   # 代价、起点坐标、路径记录
        
        visited = set()
        visited.add((start_x, start_y))  # 和BFS一样，已经被访问过了就别来了
        
        while heap:
            # 只要heap还有节点就一直循环
            cost, x, y, path = heapq.heappop(heap) # 弹出代价最低的节点
            
            if (x, y) == (target_x, target_y):
                return path                        # 因为A*规划得出的路径一定代价最低，所以到达终点直接输出路径就可以了
            
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy            # 搜索八向的邻居
                
                if (nx, ny) not in visited and self.is_valid_position(nx, ny):
                    visited.add((nx, ny))                                        # 符合条件就标记已访问
                    g = len(path) - 1                                            # 已走步数
                    h = abs(nx - target_x) + abs(ny - target_y)                  # 启发函数（曼哈顿距离，绝对值相加）
                    new_cost = g + h                                             # 总代价
                    heapq.heappush(heap, (new_cost, nx, ny, path + [(nx, ny)]))  # 将本节点加入路径，并放入队列等待搜索其邻居
        
        return None

#    def bfs_shortest_path(self, start_x, start_y, end_x, end_y):
#        # BFS寻找最短路径，用于离线训练（换成A星了）
#        if not self.map_data[end_y][end_x]:
#            return None  # 和A星不太一样的是，BFS负责学习，而非负责小车的运动，我需要的是尽可能多的路径，找到更多的路标，
#                         # 所以小车终点是否合法不在BFS的考虑范围内，只需要不在障碍物上即可。
#        
#        queue = deque([(start_x, start_y, [(start_x, start_y)])])
#        visited = set()
#        visited.add((start_x, start_y))  # flood_fill函数里已经分析过BFS了，我这代码乱了啊，懒得管了（还是管了）
#        
#        while queue:
#            x, y, path = queue.popleft()
#            
#            if x == end_x and y == end_y:
#                return path
#            
#            for dx, dy in self.directions:
#                nx, ny = x + dx, y + dy
#                if (nx, ny) not in visited and self.is_valid_position(nx, ny):  # 不在意终点，但过程还是在意的
#                    visited.add((nx, ny))                                       # is_valid_position也在后面啊（挪了）
#                    queue.append((nx, ny, path + [(nx, ny)]))
#        
#        return None
    
    def find_white_region_centers(self):
        # 找出所有白色连通区域的中心点，让离线训练知道该去哪里训练
        # 笨蛋DS居然想用BFS划分白色区域，这不纯扯淡吗，最后所有白色格子都归到同一片区域里了
        # 数字图像处理学了有用啊，我给他腐蚀了不就能中断连接了
        # 用多大的算子，太大了全封上了，太小的还连着，不过小于5*5的白色区域反正也进不去，让算子至少大于5*5吧
        # 用L型算子既能识别广场也能识别岔路
        # L型算子有四个方向，难道要处理四次？
        # 十字架怎么样
        # 我到底需要什么样子的白色区域
        # 我在选择终点啊，不是岔路口，那是路径学习的事情啊
        # 如果我用6*6的算子腐蚀，再对剩下的白色格子执行is_valid_position判定，那剩下的白色区域只能存在于大于5*5的区域中心
        # 那即便这片白色区域是条路又如何呢？这么宽给你得了
        # 干了，有什么问题运行了再说

        # 干了，训练点不均匀，要炸了呢（安详）

        # L型腐蚀，以折点为原点，只保留右方和上方都有5格连续白色的格子
        # 改3
        size = 4
        self.eroded = np.zeros((self.height, self.width), dtype=bool)  # 腐蚀图，布尔值均默认False

        for y in range(self.height):
            for x in range(self.width):
                if not self.map_data[y][x]:
                    continue                                                  # 如果是障碍就跳过本次循环
            
                # 检查向右（+x）方向
                ok_right = True
                for i in range(size):
                    if x + i >= self.width or not self.map_data[y][x + i]:    # 如果越界了或者格子是黑色的，就说明需要腐蚀
                        ok_right = False
                        break
            
                # 检查向上（-y）方向
                ok_up = True
                for i in range(size):
                    if y - i < 0 or not self.map_data[y - i][x]:
                        ok_up = False
                        break
            
                if ok_right and ok_up:
                    self.eroded[y][x] = True                                       # 符合条件，是不被腐蚀的白色格子

        visited = set()  # 记录已经处理过的格子
        centers = []     # 存储找到的中心点
        
        for y in range(self.height):
            for x in range(self.width):
                if self.eroded[y][x] and (x, y) not in visited:
                    # 遍历每个未被处理的腐蚀图白色格子
                    # 寻找腐蚀图上5*5白色区域的中心点

                    test_car = Car(x, y)
                    valid = True
                    region_cells = []

                    for cx, cy in test_car.get_cells():
                        if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                            valid = False
                            break   # 起点检查
                        if not self.eroded[cy][cx]:
                            valid = False
                            break
                        region_cells.append((cx, cy))

                    if not valid:
                        continue

                    for cx, cy in region_cells:
                        visited.add((cx, cy))

                    centers.append((x, y))

        print(f"找到 {len(centers)} 个可部署区域中心")
        return centers

    def cluster_centers(self, centers, radius=10):
        # 把距离太近的中心点合并
        if not centers:
            return []
    
        clustered = []
        used = [False] * len(centers)           # 列表乘法，长度和centers的数量一样，表示centers是否被处理过
    
        for i, (x1, y1) in enumerate(centers):
            if used[i]:                         # 如果这个点已经被归入某个簇了 
                continue                        # 跳过
        
            # 找这个点周围 radius 内的所有点
            cluster = [(x1, y1)]
            for j, (x2, y2) in enumerate(centers):
                if i != j and not used[j]:
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    if dist <= radius:
                        cluster.append((x2, y2))
                        used[j] = True
        
            # 取聚类中心（质心）
            avg_x = sum(p[0] for p in cluster) // len(cluster)
            avg_y = sum(p[1] for p in cluster) // len(cluster)
            clustered.append((avg_x, avg_y))
            used[i] = True
    
        return clustered
    
    def generate_heatmap(self, all_paths):
        # 生成路径热力图，记录每个格子被路径经过的次数
        heatmap = np.zeros((self.height, self.width))              # 创建一个全是0的大小和地图一样的二维数组
        for path in all_paths:                                     # 遍历每一条路径
            for x, y in path:                                      # 遍历这条路径上的每个格子
                if 0 <= x < self.width and 0 <= y < self.height:
                    heatmap[y][x] += 1                             # 这个格子被经过的次数+1
        return heatmap
    
    def extract_keypoints_from_heatmap(self, heatmap):
        # 从热力图提取关键点
        # 初始化
        self.key_points = []
        self.key_point_dict = {}

        print(f"热力图最大值: {np.max(heatmap)}")
        print(f"热力图平均值: {np.mean(heatmap)}")

        hot_spots = np.sum(heatmap > 3)
        print(f"热度 > 3 的格子数: {hot_spots}")
        
        # 网格化
        grid_size = 10
        grid_count = {}  #{(gx,gy): 网格累计热度值}
        
        # 统计每个网格的热度
        for y in range(self.height):
            for x in range(self.width):
                if heatmap[y][x] > 2:                                                    # 热力阈值
                    gx, gy = x // grid_size, y // grid_size                              # 网格坐标（如 x=15, grid_size=10 → gx=1）
                    grid_count[(gx, gy)] = grid_count.get((gx, gy), 0) + heatmap[y][x]   # 计算这个网格的累计热度值
        
        # 计算阈值
        if grid_count:
            threshold = max(grid_count.values()) * 0.6                                   # 阈值 = 最高热度网格*0.6
            # 提取关键点
            for (gx, gy), count in grid_count.items():
                if count > threshold:                                                    # 如果这个网格的累计热度值超过阈值，则选取这个网格
                    best_x = None
                    best_y = None
                    max_heat = 0

                    # 遍历网格取热力最高的点作为关键点
                    for dy in range(grid_size):
                        for dx in range(grid_size):
                            nx = gx * grid_size + dx
                            ny = gy * grid_size + dy
                            if (0 <= nx < self.width and 0 <= ny < self.height 
                                and self.map_data[ny][nx] 
                                and heatmap[ny][nx] > max_heat):
                                max_heat = heatmap[ny][nx]
                                best_x, best_y = nx, ny

                    if best_x is not None: 
                        kp = KeyPoint(best_x, best_y)
                        self.key_points.append(kp)
                        self.key_point_dict[(best_x, best_y)] = kp

    def deploy_car(self, x, y):
        # 部署小车
        if self.is_valid_position(x, y):   #如果(x,y)作为中心点能够部署小车就建立对象
            self.car = Car(x, y)
            self.moving = False
#            self.path_history = [(x, y)]
#            self.direction_history = []
            return True
        return False
    
    def offline_train(self):
        # 离线训练：提取关键点
        if not self.car:
            print("请先部署小车！")
            return
        
        print("开始离线训练...")

        # 获取训练点（白色区域中心）
        centers = self.find_white_region_centers()
        print(f"原始训练点: {len(centers)}")

        # 聚类合并
        if len(centers) > 50:
            centers = self.cluster_centers(centers, radius=10)
            print(f"聚类后: {len(centers)}")

        
        # 如果训练点太多就随机选50个，兼顾训练效果和训练速度
        if len(centers) > 50:
            centers = random.sample(centers, 30)
            print(f"采样到 50 个训练点")

        # 获取多个起点（随机白色格子）
        white_cells = [(x, y) for y in range(self.height)
                       for x in range(self.width)
                       if self.is_valid_position(x, y)]

        num_starts = min(5, len(white_cells))
        start_points = random.sample(white_cells, num_starts)

        # 确保包含当前小车位置
        if (self.car.x, self.car.y) not in start_points:
            start_points.append((self.car.x, self.car.y))
        
        # 从多个起点规划路径
        all_paths = []
        for start_x, start_y in start_points:    
            for i, (cx, cy) in enumerate(centers):              # enumerate()：把一个可遍历的对象（如列表）变成"带序号"的枚举对象
                print(f"起点({start_x},{start_y}) → 目标{i+1}/{len(centers)}...")
                path = self.a_star(start_x, start_y, cx, cy)
            if path:
                all_paths.append(path)
        
        if all_paths:
            heatmap = self.generate_heatmap(all_paths)
            self.extract_keypoints_from_heatmap(heatmap)
            if len(self.key_points) >= 2:
                self.build_landmark_graph()
                print(f"路标图构建完成，共 {len(self.landmark_graph)} 条路线")
            else:
                print("关键点不足2个，跳过路标图构建")

            print(f"训练完成！提取了 {len(self.key_points)} 个关键点")
            self.save_key_points()
        else:
            print("训练失败：没有找到有效路径")

    def get_region(self, x, y):
        # 获取坐标所属的区域键
        return (x // self.region_size, y // self.region_size)

    def build_landmark_graph(self):
        # 在离线训练后，预计算关键点之间的最短路径（不然小车就要自己在线规划路径了，慢死了）
        print("构建路标图...")
    
        for i, kp1 in enumerate(self.key_points):
            for j, kp2 in enumerate(self.key_points):
                if i != j:
                    key = (kp1.x, kp1.y, kp2.x, kp2.y)
                    # 使用 A* 计算路径
                    path = self.a_star(kp1.x, kp1.y, kp2.x, kp2.y)
                    if path:
                        self.landmark_graph[key] = path
                        # 同时存储反向路径
                        reverse_key = (kp2.x, kp2.y, kp1.x, kp1.y)
                        self.landmark_graph[reverse_key] = list(reversed(path))
    
        print(f"路标图构建完成，共 {len(self.landmark_graph)} 条路线")

    # DS傻乎乎的，关键点的之间的移动找一个起点和一个终点，用已知的路线找一条len最短路径就得了
    # 写了一堆奇奇怪怪的东西愣是没明白有什么用
    def find_closest_keypoint(self, x, y):
        # 找离指定坐标最近的关键点
        closest = None
        min_dist = float('inf')
    
        for kp in self.key_points:
            dist = abs(kp.x - x) + abs(kp.y - y)
            if dist < min_dist:
                min_dist = dist
                closest = kp
    
        return closest

    def _concat_paths(self, path):
        # 将关键点路径拼接成格子路径
        if len(path) < 2:
            return []
    
        result = []
        for i in range(len(path) - 1):
            k1 = path[i]
            k2 = path[i + 1]
            key = (k1.x, k1.y, k2.x, k2.y)
            if key in self.landmark_graph:
                result.extend(self.landmark_graph[key])
            else:
                print(f"警告：关键点 {k1.x},{k1.y} 到 {k2.x},{k2.y} 没有预计算路径")
                return None  # 路径不完整
    
        return result

    def find_path_via_keypoints_dijkstra(self, start_kp, end_kp):
        # Dijkstra 在关键点图上搜索（考虑路径长度）
    
        heap = [(0, start_kp, [start_kp])]  # (累计步数, 当前关键点, 路径)
        visited = {}
    
        while heap:
            dist, current_kp, path = heapq.heappop(heap)
        
            if current_kp in visited and visited[current_kp] <= dist:  # 走别的路到这个点可能路径更短
                continue
            visited[current_kp] = dist
        
            if current_kp is end_kp:
                # 拼接路径
                return self._concat_paths(path)
        
            for other_kp in self.key_points:
                if other_kp is current_kp:
                    continue
                key = (current_kp.x, current_kp.y, other_kp.x, other_kp.y)
                if key in self.landmark_graph:
                    edge_len = len(self.landmark_graph[key])
                    heapq.heappush(heap, (dist + edge_len, other_kp, path + [other_kp]))
    
        return None

    def move_car_astar(self, target_x, target_y):
        # 用A*路径移动，update调用一次移动一次
        if not self.car or not self.moving:
            return
    
        # 如果没有缓存路径，就规划一条，仅会在第一次调用时规划
        if not self.current_path:
            self.current_path = self.a_star(self.car.x, self.car.y, target_x, target_y)
            if not self.current_path:
                print("无路可走！")
                self.moving = False
                return
            
            self.current_path = self.current_path[1:]   # 去掉起点，因为小车已经在起点了
    
        # 到达终点，此时路径为空
        if not self.current_path:
            self.moving = False
            print("到达目标点！")
            
            return
    
        # 沿路径移动
        next_x, next_y = self.current_path[0]
    
        # 检查下一步是否有效
        if not self.is_valid_position(next_x, next_y):
            print("路径受阻，重新规划...")
            self.current_path = []   # 清空缓存，下次重新规划
            return
    
        self.car.x, self.car.y = next_x, next_y
        self.current_path = self.current_path[1:]    # 去掉已经走的这一步
#        self.path_history.append((next_x, next_y))
    
    def check_car_collision(self):
        # 检查小车是否与障碍物碰撞
        if not self.car:
            return False     # 小车不存在自然不会碰撞，直接返回False，防止小车不存在导致程序崩溃
        for cx, cy in self.car.get_cells():
            if not self.map_data[cy][cx]:
                return True  # 真撞了
        return False    
    
    def navigate_with_landmarks(self, target_x, target_y):
        # 分层导航
        if not self.car or not self.moving:
            return

        # 确保路标图已构建
        if len(self.key_points) >= 2 and not self.landmark_graph:
            self.build_landmark_graph()
    
        # 如果没有缓存路径，规划一条
        if not self.current_path:
            start_kp = self.find_closest_keypoint(self.car.x, self.car.y)
            target_kp = self.find_closest_keypoint(target_x, target_y)
        
            if start_kp and target_kp and start_kp is not target_kp:
                # 关键点之间的路径
                kp_path = self.find_path_via_keypoints_dijkstra(start_kp, target_kp)
                if kp_path:
                    # 第一步：从当前位置走到第一个关键点
                    path_to_first_kp = self.a_star(self.car.x, self.car.y, start_kp.x, start_kp.y)
                    if not path_to_first_kp:
                        # 走不到第一个关键点，直接 A*
                        self.move_car_astar(target_x, target_y)
                        return
                    # 拼接：走到第一个关键点 + 关键点之间路径
                    self.current_path = path_to_first_kp[1:]  # 去掉起点
                    self.current_path.extend(kp_path)
                    self.final_target = (target_x, target_y)
                    
        
            # 兜底：直接用 A*
            else:
                self.move_car_astar(target_x, target_y)
                return
    
        # 沿路径移动
        if self.current_path:
            next_x, next_y = self.current_path[0]
            if self.is_valid_position(next_x, next_y):
                self.car.x, self.car.y = next_x, next_y
                self.current_path = self.current_path[1:]
            
                if not self.current_path and self.final_target is not None:
                    tx, ty = self.final_target
                    self.move_car_astar(tx, ty)
                    self.final_target = None
            else:
                self.current_path = []
    
#    def get_car_surroundings(self, x, y):
#        # 获取小车周围6x6区域的可通行方向，本质是走一步看一步的贪心移动，想换成A*规划，但真不熟（换了）
#        available = []
#        for dx, dy in self.directions:
#            nx, ny = x + dx, y + dy
#            if self.is_valid_position(nx, ny):  
#                # 中心点八向平移，通过is_valid_position（x, y）函数直接判断会不会撞障碍物或边界
#                
#                available.append((dx, dy))
#        return available
    
#    def calculate_distance(self, x1, y1, x2, y2):
#        # 计算曼哈顿距离
#        return abs(x1 - x2) + abs(y1 - y2)  #abs输出绝对值
    
#    def get_best_direction(self, current_x, current_y, target_x, target_y, available_dirs):
#        # 获取最佳移动方向（贪心算法用的，#了也没事）  
#        if not available_dirs:
#            return None
#        
#        # 检查当前点是否为关键点
#        current_pos = (current_x, current_y)
#        if current_pos in self.key_point_dict:
#            kp = self.key_point_dict[current_pos]
#            target_key = f"{target_x},{target_y}"
#            
#            # 如果有学习到的最佳方向，优先使用
#            if target_key in kp.best_directions:
#                learned_dir = kp.best_directions[target_key]
#                # 检查学习到的方向是否可用
#                if learned_dir in available_dirs:
#                    return learned_dir
#        
#        # 计算各方向到目标的距离
#        best_dir = None
#       min_distance = float('inf')
#        
#        for dx, dy in available_dirs:
#            nx, ny = current_x + dx, current_y + dy
#            dist = self.calculate_distance(nx, ny, target_x, target_y)
#            
#            if dist < min_distance:
#                min_distance = dist
#                best_dir = (dx, dy)
#       
#        return best_dir
    
#    def detect_key_point(self, pos, old_direction, new_direction):
#        # 检测并记录关键点(依然是贪心算法的产物)
#        if pos not in self.key_point_dict:
#            kp = KeyPoint(pos[0], pos[1], 1)
#            self.key_points.append(kp)
#            self.key_point_dict[pos] = kp
#            print(f"发现新关键点: {pos}")
#       else:
#            self.key_point_dict[pos].direction_changes += 1
    
#    def move_car(self, target_x, target_y):
#        # 移动小车向目标点(贪心算法还是输给了A星)
#        if not self.car or not self.moving:
#            return
#        
#        current_x, current_y = self.car.x, self.car.y
#        
#        # 到达目标点
#        if current_x == target_x and current_y == target_y:
#           self.moving = False
#            print("到达目标点！")
#            return
#        
#        # 获取可用方向
#        available_dirs = self.get_car_surroundings(current_x, current_y)
#        
#        if not available_dirs:
#            print("无路可走！")
#            self.moving = False
#            return
#        
#        # 获取最佳方向
#        best_dir = self.get_best_direction(current_x, current_y, target_x, target_y, available_dirs)
#        
#        if not best_dir:
#            # 如果无最佳方向，选择第一个可用方向
#            best_dir = available_dirs[0]
#        # 尝试移动
#        new_x, new_y = current_x + best_dir[0], current_y + best_dir[1]
#        
#        if self.is_valid_position(new_x, new_y):
#            # 记录方向改变
#            if self.direction_history:
#                last_dir = self.direction_history[-1]
#                if last_dir != best_dir:
#                    # 方向发生改变，可能是关键点
#                    self.detect_key_point((current_x, current_y), last_dir, best_dir)
#            
#            # 执行移动
#            self.car.x, self.car.y = new_x, new_y
#            self.path_history.append((new_x, new_y))
#            self.direction_history.append(best_dir)
#        else:
#            # 移动受阻，学习此位置
#            self.detect_key_point((current_x, current_y), None, None)
    
#    def learn_from_path(self, start_x, start_y, end_x, end_y):
#        # 从走过的路径学习最优方向(依然是贪心算法的遗留物)
#        # 使用BFS找到理论最短路径
#        shortest_path = self.bfs_shortest_path(start_x, start_y, end_x, end_y)
#        
#        if not shortest_path or len(shortest_path) < 2:
#            return
#        
#        # 分析关键点
#        for i in range(len(shortest_path) - 1):
#            x, y = shortest_path[i]
#            nx, ny = shortest_path[i + 1]
#            direction = (nx - x, ny - y)
#            
#           # 检查是否是关键点
#            pos = (x, y)
#            if pos in self.key_point_dict:
#                kp = self.key_point_dict[pos]
#                target_key = f"{end_x},{end_y}"
#                
#                # 如果这个方向能更快到达目标，记录下来
#                if target_key not in kp.best_directions:
#                    kp.best_directions[target_key] = direction
#                    print(f"在关键点{pos}学习到方向{direction}可快速到达目标{end_x},{end_y}")
    
    def draw_map(self):
        # 绘制地图
        for y in range(self.height):
            for x in range(self.width):
                color = WHITE if self.map_data[y][x] else BLACK
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
    
    def draw_car(self):
        # 绘制小车
        if self.car:
            half = self.car.size // 2
            for i in range(-half, half + 1):
                for j in range(-half, half + 1):
                    x = self.car.x + i
                    y = self.car.y + j
                    if 0 <= x < self.width and 0 <= y < self.height:
                        rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                         self.cell_size, self.cell_size)
                        pygame.draw.rect(self.screen, RED, rect)
    
    def draw_target(self):
        # 绘制目标点
        if self.target:
            rect = pygame.Rect(self.target[0] * self.cell_size, 
                             self.target[1] * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, GREEN, rect)
    
    def draw_key_points(self):
        # 绘制关键点（黄色边框）
        for kp in self.key_points:
            rect = pygame.Rect(kp.x * self.cell_size, kp.y * self.cell_size,
                         self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, YELLOW, rect, 2)
    
    def draw_buttons(self):
        # 使用支持中文的字体
        try:
            font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 18) 
        except:
            font = pygame.font.Font(None, 18)
    
        # 绘制完成按钮
        pygame.draw.rect(self.screen, BLUE, self.button_rect)
        text = font.render("绘制完成", True, WHITE)
        # 动态计算居中位置
        text_rect = text.get_rect(center=self.button_rect.center)
        self.screen.blit(text, text_rect)
    
        # 清除按钮
        pygame.draw.rect(self.screen, GRAY, self.clear_button)
        text = font.render("清除地图", True, WHITE)
        text_rect = text.get_rect(center=self.clear_button.center)
        self.screen.blit(text, text_rect)
    
        # 重置按钮
        pygame.draw.rect(self.screen, GRAY, self.reset_button)
        text = font.render("重置小车", True, WHITE)
        text_rect = text.get_rect(center=self.reset_button.center)
        self.screen.blit(text, text_rect)
    
        # 训练按钮
        pygame.draw.rect(self.screen, (100, 100, 200), self.train_button)
        text = font.render("离线训练", True, WHITE)
        text_rect = text.get_rect(center=self.train_button.center)
        self.screen.blit(text, text_rect)
    
        # 画笔按钮
        brush_color = ORANGE if self.current_tool == "brush" else GRAY
        pygame.draw.rect(self.screen, brush_color, self.brush_button)
        text = font.render("画笔", True, WHITE)
        text_rect = text.get_rect(center=self.brush_button.center)
        self.screen.blit(text, text_rect)
    
        # 填充按钮
        fill_color = ORANGE if self.current_tool == "fill" else GRAY
        pygame.draw.rect(self.screen, fill_color, self.fill_button)
        text = font.render("填充", True, WHITE)
        text_rect = text.get_rect(center=self.fill_button.center)
        self.screen.blit(text, text_rect)
    
        # 状态显示（左下角，不需要居中）
        status = "绘制模式" if self.editing else ("移动模式" if not self.moving else "移动中")
        tool_text = f"当前工具: {'画笔' if self.current_tool == 'brush' else '填充'}"
        text = font.render(f"{status} | {tool_text}", True, BLACK)
        self.screen.blit(text, (10, self.window_height - 30))
    
    def handle_events(self):
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.save_key_points()
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                grid_x = x // self.cell_size
                grid_y = y // self.cell_size
                
                # 检查按钮点击
                if self.button_rect.collidepoint(x, y):
                    if self.editing:
                        self.editing = False
                        print("绘制完成，可以部署小车")
                    continue
                
                if self.clear_button.collidepoint(x, y):
                    self.map_data = np.ones((self.height, self.width), dtype=bool)
                    self.car = None
                    self.target = None
                    self.editing = True
                    continue
                
                if self.reset_button.collidepoint(x, y):
                    if self.car:
                        self.car = None
                        self.target = None
                        self.editing = True
                    continue

                # 添加训练按钮点击事件
                if self.train_button.collidepoint(x, y):
                    if self.car:
                        self.offline_train()
                    else:
                        print("请先部署小车！")
                    continue
                
                # 画笔按钮
                if self.brush_button.collidepoint(x, y):
                    self.current_tool = "brush"
                    print("切换到画笔工具")
                    continue
                
                # 填充按钮
                if self.fill_button.collidepoint(x, y):
                    self.current_tool = "fill"
                    print("切换到填充工具")
                    continue
                
                # 地图区域点击
                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    if self.editing:
                        # 编辑模式
                        if self.current_tool == "brush":
                            # 画笔工具：绘制障碍物
                            self.drawing = True
                            self.map_data[grid_y][grid_x] = False
                        elif self.current_tool == "fill":
                            # 填充工具：泛洪填充
                            # 左键填充障碍物，右键填充可通行区域
                            if event.button == 1:  # 左键
                                self.flood_fill(grid_x, grid_y, BLACK)
                            elif event.button == 3:  # 右键
                                self.flood_fill(grid_x, grid_y, WHITE)
                    
                    elif not self.moving and self.car:
                        # 移动模式：设置目标点
                        if self.map_data[grid_y][grid_x]:  # 目标点必须在白色区域
                            self.target = (grid_x, grid_y)
                            self.moving = True
                            print(f"设置目标点: ({grid_x}, {grid_y})")
                            
                            
                        else:
                            print("目标点必须在可行走区域！")
                    
                    elif not self.car and not self.editing:
                        # 部署小车
                        if self.deploy_car(grid_x, grid_y):
                            print(f"小车已部署在 ({grid_x}, {grid_y})")
                        else:
                            print("部署失败：位置无效或与障碍物重叠！")
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.drawing = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.drawing and self.editing and self.current_tool == "brush":
                    x, y = event.pos
                    grid_x = x // self.cell_size
                    grid_y = y // self.cell_size
                    if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                        self.map_data[grid_y][grid_x] = False
        
        return True
    
    def update(self):
        # 更新游戏状态
        current_time = pygame.time.get_ticks()
        
        if self.moving and self.target and self.car:
            if current_time - self.last_move_time >= self.move_interval:
                self.navigate_with_landmarks(self.target[0], self.target[1])
                self.last_move_time = current_time
    
    def run(self):
        # 主循环
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.handle_events()
            self.update()
            
            # 绘制
            self.screen.fill(WHITE)
            self.draw_map()
            self.draw_key_points()
            self.draw_car()
            self.draw_target()
            self.draw_buttons()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

def main():
    # 主函数
    print("=== 智能小车导航系统 ===")
    print("工具说明：")
    print("- 画笔：左键绘制黑色障碍物（孩子们我会断触）")
    print("- 填充：左键填充黑色区域（孩子们边缘不够厚就等着全黑吧）")
    print("- 填充：右键填充白色区域（孩子们我是橡皮擦）")
    print("- 按住鼠标拖动可连续绘制")
    
    # 1. 先获取地图名
    map_name = input("请给这张地图起个名字: ").strip()
    if not map_name:
        map_name = "default"

    # 2. 检查是否有存档
    save_dir = "saves"
    save_file = os.path.join(save_dir, f"key_points_{map_name}.json")

    if os.path.exists(save_file):
        # 有存档，读取尺寸
        with open(save_file, 'r') as f:
            info = json.load(f)
            width = info["width"]
            height = info["height"]
        print(f"加载已有地图: {map_name} ({width}x{height})")
    else:
        # 无存档，询问尺寸
        print("创建新地图...")
        while True:
            try:
                width = int(input("请输入画布宽度 (100-300): "))
                height = int(input("请输入画布高度 (100-300): "))
                if 100 <= width <= 300 and 100 <= height <= 300:
                    break
                else:
                    print("尺寸必须在100-300之间！")
            except ValueError:
                print("请输入有效的数字！")
    
    # 3. 创建并运行程序
    app = MapNavigation(width, height, map_name)
    app.run()

if __name__ == "__main__":
    main()      # 直接运行这个.py文件才会执行 main()