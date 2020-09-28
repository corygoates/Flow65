from airfoil_tool import VortexPanelAirfoil

if __name__=="__main__":

    for NACA in ["2421", "0015"]:
       # Load airfoil
        airfoil = VortexPanelAirfoil(NACA=NACA,
                                     x_le=0.0,
                                     x_te=1.0)

        airfoil.panel(99)

        # Loop through alpha
        V = 1.0
        for a in [-5.0, 0.0, 5.0]:

            # Solve
            airfoil.set_condition(alpha=a, V=V)
            airfoil.solve()

            # Plot
            airfoil.plot(-1.5, [-1.5, 1.5], 0.01, 20, 0.0625)
