import heapq
import math
from collections import defaultdict, deque

# Классы из методички
class Problem:
    """Абстрактный класс для формальной задачи."""
    def __init__(self, initial=None, goal=None, **kwds):
        self.__dict__.update(initial=initial, goal=goal, **kwds)
    
    def actions(self, state):
        raise NotImplementedError
    
    def result(self, state, action):
        raise NotImplementedError
    
    def is_goal(self, state):
        return state == self.goal
    
    def action_cost(self, s, a, s1):
        return 1
    
    def h(self, node):
        return 0
    
    def __str__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.initial, self.goal)


class Node:
    """Узел в дереве поиска"""
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action,
                           path_cost=path_cost)
    
    def __repr__(self):
        return '<{}>'.format(self.state)
    
    def __len__(self):
        return 0 if self.parent is None else (1 + len(self.parent))
    
    def __lt__(self, other):
        return self.path_cost < other.path_cost


failure = Node('failure', path_cost=math.inf)
cutoff = Node('cutoff', path_cost=math.inf)


def expand(problem, node):
    """Раскрываем узел, создав дочерние узлы."""
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node):
    """Последовательность действий, чтобы добраться до этого узла."""
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    """Последовательность состояний, чтобы добраться до этого узла."""
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]


class PriorityQueue:
    """Очередь с приоритетом"""
    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []  # heap of (score, item) pairs
        for item in items:
            self.add(item)
    
    def add(self, item):
        """Добавляем элемент в очередь."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)
    
    def pop(self):
        """Достаем и возвращаем элемент с минимальным значением f(item)."""
        return heapq.heappop(self.items)[1]
    
    def top(self):
        return self.items[0][1]
    
    def __len__(self):
        return len(self.items)


# Граф дорог Румынии (как в примере)
romania_map = {
    'Arad': [('Zerind', 75), ('Sibiu', 140), ('Timisoara', 118)],
    'Zerind': [('Arad', 75), ('Oradea', 71)],
    'Oradea': [('Zerind', 71), ('Sibiu', 151)],
    'Sibiu': [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu Vilcea', 80)],
    'Timisoara': [('Arad', 118), ('Lugoj', 111)],
    'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
    'Mehadia': [('Lugoj', 70), ('Drobeta', 75)],
    'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
    'Craiova': [('Drobeta', 120), ('Rimnicu Vilcea', 146), ('Pitesti', 138)],
    'Rimnicu Vilcea': [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Pitesti': [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)],
    'Bucharest': [('Fagaras', 211), ('Pitesti', 101), ('Giurgiu', 90), ('Urziceni', 85)],
    'Giurgiu': [('Bucharest', 90)],
    'Urziceni': [('Bucharest', 85), ('Vaslui', 142), ('Hirsova', 98)],
    'Hirsova': [('Urziceni', 98), ('Eforie', 86)],
    'Eforie': [('Hirsova', 86)],
    'Vaslui': [('Urziceni', 142), ('Iasi', 92)],
    'Iasi': [('Vaslui', 92), ('Neamt', 87)],
    'Neamt': [('Iasi', 87)]
}

# Прямолинейные расстояния до Бухареста (эвристика)
straight_line_to_bucharest = {
    'Arad': 366,
    'Bucharest': 0,
    'Craiova': 160,
    'Drobeta': 242,
    'Eforie': 161,
    'Fagaras': 176,
    'Giurgiu': 77,
    'Hirsova': 151,
    'Iasi': 226,
    'Lugoj': 244,
    'Mehadia': 241,
    'Neamt': 234,
    'Oradea': 380,
    'Pitesti': 100,
    'Rimnicu Vilcea': 193,
    'Sibiu': 253,
    'Timisoara': 329,
    'Urziceni': 80,
    'Vaslui': 199,
    'Zerind': 374
}


class GraphProblem(Problem):
    """Задача поиска пути на графе."""
    def __init__(self, initial, goal, graph):
        super().__init__(initial=initial, goal=goal)
        self.graph = graph
    
    def actions(self, state):
        """Возвращает возможные действия из состояния."""
        return [city for city, _ in self.graph.get(state, [])]
    
    def result(self, state, action):
        """Возвращает новое состояние после выполнения действия."""
        return action
    
    def action_cost(self, s, a, s1):
        """Возвращает стоимость действия."""
        # Ищем стоимость перехода из s в s1
        for city, cost in self.graph[s]:
            if city == s1:
                return cost
        return math.inf
    
    def h(self, node):
        """Эвристическая функция - прямолинейное расстояние до цели."""
        return straight_line_to_bucharest.get(node.state, math.inf)


def best_first_search(problem, f):
    """Поиск по первому наилучшему совпадению."""
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    
    return failure


def astar_search(problem):
    """Поиск A* - f(n) = g(n) + h(n)."""
    return best_first_search(problem, f=lambda n: n.path_cost + problem.h(n))


def main():
    """Основная функция для поиска пути от Арада до Бухареста."""
    print("Поиск оптимального пути от Арада до Бухареста")
    print("=" * 50)
    
    # Создаем задачу
    problem = GraphProblem('Arad', 'Bucharest', romania_map)
    
    # Выполняем поиск A*
    print("\nВыполняем поиск A*...")
    solution = astar_search(problem)
    
    if solution == failure:
        print("Путь не найден!")
    else:
        # Получаем путь
        path = path_states(solution)
        actions = path_actions(solution)
        total_cost = solution.path_cost
        
        # Выводим результаты
        print(f"\nНайден оптимальный путь!")
        print(f"Количество городов: {len(path)}")
        print(f"Общая дистанция: {total_cost} км")
        print(f"Общая стоимость: {total_cost}")
        
        print("\nМаршрут:")
        for i, (city, next_city) in enumerate(zip(path, path[1:] + [''])):
            if next_city:
                # Находим стоимость перехода
                cost = None
                for c, cst in romania_map[city]:
                    if c == next_city:
                        cost = cst
                        break
                print(f"{i+1}. {city} -> {next_city} ({cost} км)")
            else:
                print(f"{i+1}. {city} (цель достигнута!)")
        
        print(f"\nПуть: {' -> '.join(path)}")
        
        # Проверяем корректность пути (для примера из литературы)
        expected_path = ['Arad', 'Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest']
        if path == expected_path:
            print("\n✓ Найден ожидаемый оптимальный путь!")
        else:
            print(f"\nПримечание: ожидался путь {expected_path}")
    
    return solution


if __name__ == "__main__":
    # Запускаем поиск
    result = main()
    
    # Дополнительная информация
    print("\n" + "=" * 50)
    print("Информация о графе:")
    print(f"Всего городов: {len(romania_map)}")
    
    # Подсчет всех дорог
    total_roads = sum(len(connections) for connections in romania_map.values())
    print(f"Всего дорог: {total_roads // 2} (каждая дорога учтена дважды)")