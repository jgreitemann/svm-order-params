set terminal cairolatex pdf
set output 'Td.tex'

set xlabel '$J_1 = J_3$'
plot 'Td.test.txt' using 2:5:6 with yerrorlines lw 1.5 title '$T_d$ decision function'