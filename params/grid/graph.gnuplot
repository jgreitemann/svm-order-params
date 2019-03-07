set terminal cairolatex pdf
set output 'graph.tex'

set cbtics 0.02

set palette defined ( -0.08 '#D53E4F',\
                      -0.07 '#F46D43',\
                      -0.065 '#FDAE61',\
                      -0.06 '#FEE08B',\
                      -0.03 '#E6F598',\
                       0.04 '#ABDDA4',\
                       0.10 '#66C2A5',\
                       0.16 '#3288BD' )

plot 'edges.txt' using 2:1 with lines notitle, \
     'phases.txt' index 1 using 2:1:3 with points pt 7 lc palette notitle