import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random
import math
from OpenGL.arrays import vbo
import ctypes

# --- CONFIG ---
ANCHO_VENTANA = 1024
ALTO_VENTANA = 768
TAMANO_MALLA = 80   # más detalle
ESCALA = 0.8
TITLE_BAR_HEIGHT = 44  # altura de la barra de título (pixeles)

def generar_terreno(size=TAMANO_MALLA, escala=ESCALA):
    """Genera una malla de alturas y colores (centered)."""
    h = np.zeros((size, size), dtype=float)
    cols = np.zeros((size, size, 3), dtype=float)
    center = (size - 1) / 2.0

    for i in range(size):
        for j in range(size):
            x = (i - center) * escala
            z = (j - center) * escala

            # mezcla de varias funciones para más realismo
            distancia = math.hypot(x, z)
            base = 3.0 * math.exp(- (distancia / (size*0.12))**2)      # colina central
            ruido = random.uniform(-0.8, 1.2) * (1.0 - j/size)         # más irregular hacia una zona
            acantilado = max(0.0, (x + size*escala*0.2) * 0.6) if x > 0 else -2.5  # costa a la izquierda

            if x < -size*escala*0.08:
                py = -2.0 + random.uniform(-0.1, 0.2)  # mar
                color = np.array([0.05, 0.18, 0.4])
            else:
                py = base + ruido + acantilado * 0.03 + math.sin(z*0.3) * 0.8
                r = 0.6 + random.uniform(-0.08, 0.08)
                g = 0.45 + random.uniform(-0.08, 0.06)
                b = 0.2 + random.uniform(-0.05, 0.04)
                color = np.array([r, g, b])

            h[i, j] = py
            cols[i, j] = color

    # generar vértices centrados
    verts = np.zeros((size, size, 3), dtype=float)
    for i in range(size):
        for j in range(size):
            verts[i, j] = np.array([ (i-center)*escala, h[i,j], (j-center)*escala ])
    return verts, cols

def calcular_normales(verts):
    """Calcula normales por vértice promediando normales de caras adyacentes."""
    size_x, size_z, _ = verts.shape
    normales = np.zeros_like(verts)
    for i in range(size_x - 1):
        for j in range(size_z - 1):
            v0 = verts[i  , j  ]
            v1 = verts[i+1, j  ]
            v2 = verts[i+1, j+1]
            v3 = verts[i  , j+1]
            # dos triángulos por quad
            n1 = np.cross(v1 - v0, v2 - v0)
            n2 = np.cross(v2 - v0, v3 - v0)
            for idx in [(i,j),(i+1,j),(i+1,j+1)]:
                normales[idx] += n1
            for idx in [(i,j),(i+1,j+1),(i,j+1)]:
                normales[idx] += n2
    # normalizar
    norms = np.linalg.norm(normales, axis=2)
    norms[norms == 0] = 1.0
    normales /= norms[:,:,np.newaxis]
    return normales

def dibujar_terreno(verts, cols, normales):
    size_x, size_z, _ = verts.shape
    glBegin(GL_TRIANGLES)
    for i in range(size_x - 1):
        for j in range(size_z - 1):
            # tri 1 (v0, v1, v2)
            for vi in [(i,j),(i+1,j),(i+1,j+1)]:
                glColor3fv(cols[vi])
                glNormal3fv(normales[vi])
                glVertex3fv(verts[vi])
            # tri 2 (v0, v2, v3)
            for vi in [(i,j),(i+1,j+1),(i,j+1)]:
                glColor3fv(cols[vi])
                glNormal3fv(normales[vi])
                glVertex3fv(verts[vi])
    glEnd()

def crear_vbos(verts, cols, normales):
    """
    Recibe verts (NxNx3), cols (NxNx3), normales (NxNx3).
    Devuelve (vbo_v, vbo_n, vbo_c, indices, index_count)
    """
    size_x, size_z, _ = verts.shape
    # Flatten vertices, normals, colors
    vert_list = verts.reshape((-1, 3)).astype(np.float32)
    norm_list = normales.reshape((-1, 3)).astype(np.float32)
    col_list  = cols.reshape((-1, 3)).astype(np.float32)

    # Build triangle indices (two triangles por quad)
    indices = []
    def idx(i, j): return i * size_z + j
    for i in range(size_x - 1):
        for j in range(size_z - 1):
            # tri 1
            indices.append(idx(i, j))
            indices.append(idx(i+1, j))
            indices.append(idx(i+1, j+1))
            # tri 2
            indices.append(idx(i, j))
            indices.append(idx(i+1, j+1))
            indices.append(idx(i, j+1))
    indices = np.array(indices, dtype=np.uint32)

    # Create VBOs
    vbo_v = vbo.VBO(vert_list)
    vbo_n = vbo.VBO(norm_list)
    vbo_c = vbo.VBO(col_list)
    # bind now (will rebind on draw)
    return vbo_v, vbo_n, vbo_c, indices, indices.size

def dibujar_terreno_vbo(vbo_v, vbo_n, vbo_c, indices, index_count):
    # Enable client arrays
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

    # Bind vertex VBO
    vbo_v.bind()
    glVertexPointer(3, GL_FLOAT, 0, vbo_v)
    # normals
    vbo_n.bind()
    glNormalPointer(GL_FLOAT, 0, vbo_n)
    # colors
    vbo_c.bind()
    glColorPointer(3, GL_FLOAT, 0, vbo_c)

    # Draw elements from client-side index array
    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, indices)

    # Unbind
    vbo_v.unbind()
    vbo_n.unbind()
    vbo_c.unbind()

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

def configurar_iluminacion():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION,  (50.0, 80.0, 50.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,   (1.0, 1.0, 0.95, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR,  (1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT,   (0.12, 0.12, 0.12, 1.0))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.2,0.2,0.2,1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 16.0)

# --- UI botones ---
BUTTON_SIZE = 34
BUTTON_PADDING = 8

# --- estado de la ventana ---
is_maximized = False
prev_size = (ANCHO_VENTANA, ALTO_VENTANA)

def ajustar_ventana(w, h):
    """Recrea la ventana y ajusta viewport/proyección OpenGL."""
    global prev_size, is_maximized
    # si no estamos maximizando, guardamos tamaño previo
    if not is_maximized:
        prev_size = (w, h)
    pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL | RESIZABLE)
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Use full-window projection aquí; en el bucle se ajusta la proyección de la escena
    gluPerspective(45.0, (w / float(max(1, h))), 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Restaurar estados OpenGL necesarios después de set_mode
    glClearColor(0.0, 0.0, 0.05, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)
    configurar_iluminacion()

def dibujar_ui(w, h):
    """Dibuja barra de título en la parte superior y 3 botones alineados a la derecha dentro de la barra.
    Si la ventana está maximizada usa el ancho del monitor como límite de la barra."""
    spacing = 8  # espacio entre botones
    info = pygame.display.Info()
    monitor_w = info.current_w

    # si la ventana coincide con el monitor (maximizada) usamos el ancho del monitor
    bar_width = monitor_w if w >= monitor_w else w

    total_width = 3 * BUTTON_SIZE + 2 * spacing
    # Posición X inicial de la primera (más a la izquierda de los tres) dentro de la barra, alineado a la derecha
    draw_right = min(bar_width, w)
    start_x = draw_right - BUTTON_PADDING - total_width
    bar_y_top = h
    bar_y_bottom = h - TITLE_BAR_HEIGHT

    # Preparar proyección 2D (coordenadas en la ventana)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, w, 0, h, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)

    # Dibujar barra de título (fondo).
    glColor3f(0.12, 0.12, 0.16)
    glBegin(GL_QUADS)
    glVertex2f(0, bar_y_bottom)
    glVertex2f(draw_right, bar_y_bottom)
    glVertex2f(draw_right, bar_y_top)
    glVertex2f(0, bar_y_top)
    glEnd()

    # Posiciones botones (derecha, uno seguido del otro) - ajustadas al límite usado (draw_right)
    sx = start_x
    mx = start_x + (BUTTON_SIZE + spacing)
    cx = start_x + 2 * (BUTTON_SIZE + spacing)
    cy = bar_y_bottom + (TITLE_BAR_HEIGHT - BUTTON_SIZE) / 2.0  # centrar verticalmente en la barra

    # Dibujar botones (recortamos si están fuera de la ventana)
    def draw_button(x, y, color, draw_icon):
        if x + BUTTON_SIZE < 0 or x > w:
            return
        glColor3fv(color)
        glBegin(GL_QUADS)
        glVertex2f(x,   y)
        glVertex2f(x+BUTTON_SIZE, y)
        glVertex2f(x+BUTTON_SIZE, y+BUTTON_SIZE)
        glVertex2f(x,   y+BUTTON_SIZE)
        glEnd()
        draw_icon()

    # Minimizar
    draw_button(sx, cy, (0.95,0.85,0.18), lambda: (
        glColor3f(0.05,0.05,0.05),
        glLineWidth(3.0),
        glBegin(GL_LINES),
        glVertex2f(sx+6, cy+BUTTON_SIZE/2.0), glVertex2f(sx+BUTTON_SIZE-6, cy+BUTTON_SIZE/2.0),
        glEnd()
    ))

    # Maximizar
    draw_button(mx, cy, (0.15,0.7,0.15), lambda: (
        glColor3f(1,1,1),
        glBegin(GL_LINE_LOOP),
        glVertex2f(mx+6, cy+6), glVertex2f(mx+BUTTON_SIZE-6, cy+6),
        glVertex2f(mx+BUTTON_SIZE-6, cy+BUTTON_SIZE-6), glVertex2f(mx+6, cy+BUTTON_SIZE-6),
        glEnd()
    ))

    # Cerrar
    draw_button(cx, cy, (0.85,0.12,0.12), lambda: (
        glColor3f(1,1,1),
        glLineWidth(2.5),
        glBegin(GL_LINES),
        glVertex2f(cx+6, cy+6), glVertex2f(cx+BUTTON_SIZE-6, cy+BUTTON_SIZE-6),
        glVertex2f(cx+BUTTON_SIZE-6, cy+6), glVertex2f(cx+6, cy+BUTTON_SIZE-6),
        glEnd()
    ))

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    # Restaurar matrices
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def ui_hit_test(pos, w, h):
    """Devuelve 'min', 'max', 'close' o None según la posición del click.
    Interpreta coords de pygame (y desde arriba). Usa el ancho del monitor si la ventana está maximizada."""
    x, y = pos  # pygame coords (0,0) arriba-izquierda
    spacing = 8
    info = pygame.display.Info()
    monitor_w = info.current_w

    bar_width = monitor_w if w >= monitor_w else w
    draw_right = min(bar_width, w)
    total_width = 3 * BUTTON_SIZE + 2 * spacing
    start_x = draw_right - BUTTON_PADDING - total_width
    sx = start_x
    mx = start_x + (BUTTON_SIZE + spacing)
    cx = start_x + 2 * (BUTTON_SIZE + spacing)

    # botón Y en coords pygame (distancia desde top)
    btn_y = (TITLE_BAR_HEIGHT - BUTTON_SIZE) / 2.0
    rects = {
        'min': (sx, btn_y, BUTTON_SIZE, BUTTON_SIZE),
        'max': (mx, btn_y, BUTTON_SIZE, BUTTON_SIZE),
        'close': (cx, btn_y, BUTTON_SIZE, BUTTON_SIZE)
    }
    for name, (rx, ry, rw_rect, rh_rect) in rects.items():
        if rx <= x <= rx + rw_rect and ry <= y <= ry + rh_rect:
            return name
    return None

# --- Acciones de los botones ---
def accion_minimizar():
    """Minimiza la ventana (iconify)."""
    # minimizar a barra y no cambiar tamaño
    try:
        pygame.display.iconify()
    except Exception:
        # fallback: redimensionar a la mitad si iconify no funciona
        info = pygame.display.Info()
        ajustar_ventana(max(200, info.current_w // 2), max(200, info.current_h // 2))

def accion_maximizar():
    """Toggle maximizar/restaurar al tamaño previo."""
    global is_maximized, prev_size
    info = pygame.display.Info()
    if not is_maximized:
        # guardar tamaño actual y maximizar al tamaño del monitor
        prev_size = pygame.display.get_surface().get_size()
        ajustar_ventana(info.current_w, info.current_h)
        is_maximized = True
    else:
        # restaurar tamaño previo
        w, h = prev_size
        ajustar_ventana(w, h)
        is_maximized = False

def accion_cerrar():
    """Envía evento de QUIT para cerrar limpiamente."""
    pygame.event.post(pygame.event.Event(QUIT))

def calcular_borde_urbano_mina(verts, cols):
    """
    Calcula una lista de puntos 3D (x,y,z) que aproximan la línea
    frontera entre zonas 'urbanas' y 'mina'. Clasificación simple:
    se considera 'mina' si r > g + umbral.
    Se busca la transición por cada fila z y se interpola entre
    dos vértices para mejorar la continuidad.
    """
    size_x, size_z, _ = verts.shape
    umbral = 0.02
    puntos = []

    for j in range(size_z):
        prev_is_mine = None
        for i in range(size_x):
            r,g,b = cols[i, j]
            is_mine = (r > g + umbral)
            if prev_is_mine is None:
                prev_is_mine = is_mine
                continue
            if is_mine != prev_is_mine:
                # interpolación entre (i-1,j) y (i,j)
                v1 = verts[i-1, j]
                v2 = verts[i, j]
                c1 = cols[i-1, j]
                c2 = cols[i, j]
                a1 = c1[0] - c1[1]
                a2 = c2[0] - c2[1]
                if (a2 - a1) != 0:
                    t = (umbral - a1) / (a2 - a1)
                    t = float(np.clip(t, 0.0, 1.0))
                else:
                    t = 0.5
                p = v1 * (1.0 - t) + v2 * t
                # elevar ligeramente la línea sobre la superficie para que no z-fightee
                p[1] += 0.06
                puntos.append(p)
                break
    if len(puntos) == 0:
        return np.zeros((0,3), dtype=np.float32)
    return np.array(puntos, dtype=np.float32)

def dibujar_borde_rojo(puntos):
    """Dibuja la línea roja (polilínea) que marca el borde."""
    if puntos is None or puntos.size == 0:
        return
    glDisable(GL_LIGHTING)
    glEnable(GL_LINE_SMOOTH)
    glLineWidth(3.5)
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINE_STRIP)
    for p in puntos:
        glVertex3fv(p)
    glEnd()
    glLineWidth(1.0)
    glDisable(GL_LINE_SMOOTH)
    glEnable(GL_LIGHTING)

def main():
    pygame.init()
    screen = (ANCHO_VENTANA, ALTO_VENTANA)
    pygame.display.set_mode(screen, DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("Relieve 3D - OpenGL (Marcona estilo)")

    # Estado OpenGL inicial correcto
    glClearColor(0.0, 0.0, 0.05, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)
    # Configurar proyección inicial y activar iluminación
    ajustar_ventana(screen[0], screen[1])

    verts, cols = generar_terreno()
    normales = calcular_normales(verts)

    # Crear VBOs UNA vez (reduce el trabajo por frame)
    vbo_v, vbo_n, vbo_c, indices, index_count = crear_vbos(verts, cols, normales)

    # calcular la línea frontera una sola vez
    borde_rojo = calcular_borde_urbano_mina(verts, cols)

    # cámara tipo orbit
    distancia = 90.0
    yaw = 45.0   # rotY
    pitch = 35.0 # rotX
    target = np.array([0.0, 0.0, 0.0])

    mouse_down = False
    last_mouse = (0,0)

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                # usuario redimensionó la ventana manualmente
                w, h = event.size
                ajustar_ventana(w, h)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    w, h = pygame.display.get_surface().get_size()
                    hit = ui_hit_test(event.pos, w, h)
                    if hit is not None:
                        if hit == 'min':
                            accion_minimizar()
                        elif hit == 'max':
                            accion_maximizar()
                        elif hit == 'close':
                            accion_cerrar()
                        continue  # si fue click en la barra no se inicia drag de escena
                    # si no fue en UI, iniciar interacción con la escena
                    mouse_down = True
                    last_mouse = event.pos
                elif event.button == 4:  # scroll up
                    distancia = max(10.0, distancia - 5.0)
                elif event.button == 5:  # scroll down
                    distancia += 5.0
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            elif event.type == MOUSEMOTION and mouse_down:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                yaw += dx * 0.3
                pitch = max(-89.0, min(89.0, pitch + dy * 0.2))
                last_mouse = event.pos

        # calcular cámara
        cam_x = target[0] + distancia * math.cos(math.radians(pitch)) * math.sin(math.radians(yaw))
        cam_y = target[1] + distancia * math.sin(math.radians(pitch))
        cam_z = target[2] + distancia * math.cos(math.radians(pitch)) * math.cos(math.radians(yaw))

        # obtener tamaño actual de ventana para UI y viewport si necesario
        win_w, win_h = pygame.display.get_surface().get_size()

        # --- configurar viewport/proyección solo para la escena (debajo de la barra) ---
        scene_h = max(100, win_h - TITLE_BAR_HEIGHT)
        glViewport(0, 0, win_w, scene_h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, (win_w / float(scene_h)), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gluLookAt(cam_x, cam_y, cam_z, target[0], target[1], target[2], 0,1,0)

        # dibujar escena como antes (no dibuja sobre la barra)
        # pequeña cuadrícula de referencia
        glDisable(GL_LIGHTING)
        glColor3f(0.6,0.6,0.6)
        glBegin(GL_LINES)
        step = 5
        ext = 50
        for i in range(-ext, ext+1, step):
            glVertex3f(i, -3.0, -ext)
            glVertex3f(i, -3.0, ext)
            glVertex3f(-ext, -3.0, i)
            glVertex3f(ext, -3.0, i)
        glEnd()
        glEnable(GL_LIGHTING)

        dibujar_terreno_vbo(vbo_v, vbo_n, vbo_c, indices, index_count)

        # dibujar la línea roja que separa urbano / mina
        dibujar_borde_rojo(borde_rojo)

        # plano de agua translúcido
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.0, 0.35, 0.7, 0.4)
        glBegin(GL_QUADS)
        glVertex3f(-120, -1.5, -120)
        glVertex3f(120, -1.5, -120)
        glVertex3f(120, -1.5, 120)
        glVertex3f(-120, -1.5, 120)
        glEnd()
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

        # --- Restaurar viewport completo para dibujar la UI (barra de título) ---
        glViewport(0, 0, win_w, win_h)

        dibujar_ui(win_w, win_h)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()