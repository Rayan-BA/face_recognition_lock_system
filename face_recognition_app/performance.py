import pstats
p = pstats.Stats('why.txt')
p.sort_stats('tottime').print_stats(10)
p.sort_stats('cumtime').print_stats(10)