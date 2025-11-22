import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random

# --- CONFIGURACIÓN DE LA SIMULACIÓN ---
ANCHO_VENTANA = 800
ALTO_VENTANA = 600
TAMANO_MALLA = 60  # Tamaño del grid (60x60)
ESCALA = 1.0       # Separación entre puntos

# --- GENERACIÓN DE DATOS DEL RELIEVE (Simulando Marcona) ---
def generar_relieve_marcona():
    """
    Genera vértices y colores simulando la geografía de Marcona:
    - Zona de mar (plana, azul).
    - Zona de acantilado (subida abrupta).
    - Zona de desierto/mina (irregular, rojiza por el hierro).
    """
    vertices = []
    colores = []
    
    # Generamos una grilla x, z
    for x in range(TAMANO_MALLA):
        fila_v = []
        fila_c = []
        for z in range(TAMANO_MALLA):
            
            # Coordenadas base centradas
            px = (x - TAMANO_MALLA / 2) * ESCALA
            pz = (z - TAMANO_MALLA / 2) * ESCALA
            
            # Lógica de altura (py = altura)
            # Simulamos la costa: Si x < 10 es mar, x > 10 es tierra
            limite_costa = 0 
            
            if px < limite_costa:
                # ES EL MAR
                py = -2.0 + (random.uniform(-0.1, 0.1)) # Pequeño oleaje
                # Color Azul oscuro/verdoso
                color = (0.1, 0.3, 0.6) 
            else:
                # ES TIERRA (Marcona: Desierto, Hierro, Acantilado)
                
                # Factor de acantilado (subida rápida cerca del límite)
                distancia_borde = px - limite_costa
                altura_base = 5.0
                
                # Ruido para simular terreno rocoso
                ruido = random.uniform(-0.5, 1.5)
                
                # Si está muy cerca del borde, hacemos la subida (acantilado)
                if distancia_borde < 5:
                    py = (distancia_borde * 1.5) + ruido
                else:
                    # La meseta desértica
                    py = altura_base + ruido + (np.sin(pz/5.0)*1.0)

                # Color Rojizo/Marrón (Hierro y desierto)
                r = 0.6 + random.uniform(-0.05, 0.05) # Rojo predominante
                g = 0.4 + random.uniform(-0.05, 0.05)
                b = 0.2 + random.uniform(-0.05, 0.05)
                color = (r, g, b)

            fila_v.append((px, py, pz))
            fila_c.append(color)
            
        vertices.append(fila_v)
        colores.append(fila_c)
        
    return vertices, colores

def dibujar_terreno(vertices, colores):
    glBegin(GL_QUADS)
    for x in range(TAMANO_MALLA - 1):
        for z in range(TAMANO_MALLA - 1):
            # Obtenemos los 4 puntos del cuadrado (Quad)
            p1 = vertices[x][z]
            p2 = vertices[x+1][z]
            p3 = vertices[x+1][z+1]
            p4 = vertices[x][z+1]
            
            # Usamos el color del primer vértice para el quad
            c = colores[x][z]
            glColor3fv(c)
            
            glVertex3fv(p1)
            glVertex3fv(p2)
            glVertex3fv(p3)
            glVertex3fv(p4)
    glEnd()

def main():
    pygame.init()
    display = (ANCHO_VENTANA, ALTO_VENTANA)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Simulación Relieve Marcona - OpenGL")

    # Configuración de la cámara (Perspectiva)
    gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
    glTranslatef(0.0, -5.0, -40) # Mover la cámara atrás y arriba
    glRotatef(45, 1, 0, 0)       # Inclinar para ver desde arriba

    # Generar los datos una sola vez
    verts, cols = generar_relieve_marcona()

    rot_x = 0
    rot_y = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # Control de teclado simple
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    rot_y -= 5
                if event.key == pygame.K_RIGHT:
                    rot_y += 5
                if event.key == pygame.K_UP:
                    rot_x -= 5
                if event.key == pygame.K_DOWN:
                    rot_x += 5
                if event.key == pygame.K_z: # Zoom in
                    glTranslatef(0,0,1)
                if event.key == pygame.K_x: # Zoom out
                    glTranslatef(0,0,-1)

        # Limpiar pantalla
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST) # Activar profundidad para que no se vea transparente

        # Guardar matriz actual
        glPushMatrix()
        
        # Aplicar rotaciones del usuario
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)

        # Dibujar
        dibujar_terreno(verts, cols)
        
        # Dibujar "Mar" plano semi-transparente (opcional, nivel del agua)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.0, 0.4, 0.8, 0.5)
        glBegin(GL_QUADS)
        glVertex3f(-50, -1.5, -50)
        glVertex3f(50, -1.5, -50)
        glVertex3f(50, -1.5, 50)
        glVertex3f(-50, -1.5, 50)
        glEnd()
        glDisable(GL_BLEND)

        # Restaurar matriz
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()