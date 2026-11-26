"""PIA: cálculo tensorial en R². Ejecutar: python -m streamlit run aplicacion_tensorial.py
Prueba reproducible sin Streamlit: python aplicacion_tensorial.py --demo
Las expresiones simbólicas son entradas locales de confianza, no código de terceros.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

x, y, theta = sp.symbols('x y theta', real=True)
r = sp.symbols('r', positive=True)
DEFAULT = 'x**2 + 2*y**2 + x*y'
CASES = [(1.,1.,2.,1.),(-2.,1.,-1.,2.),(1.,-2.,3.,-1.)]


def model(expr, polar=False):
    symbols = (r,theta) if polar else (x,y)
    f = sp.sympify(expr, locals={'x':x,'y':y,'r':r,'theta':theta,'pi':sp.pi}, evaluate=True)
    if not isinstance(f,sp.Expr) or not f.free_symbols <= set(symbols):
        raise ValueError('Use solo las variables del sistema elegido y una expresión escalar.')
    if polar:
        F = f
        f = F.subs({r:sp.sqrt(x*x+y*y),theta:sp.atan2(y,x)}, simultaneous=True)
    else:
        F = f.subs({x:r*sp.cos(theta),y:r*sp.sin(theta)}, simultaneous=True)
    dc = tuple(sp.diff(f,q) for q in (x,y))
    dp = tuple(sp.diff(F,q) for q in (r,theta))
    return dict(f=f,F=F,dc=dc,dp=dp, fc=sp.lambdify((x,y),f,'numpy'),
                fp=sp.lambdify((r,theta),F,'numpy'),
                dcf=sp.lambdify((x,y),dc,'numpy'),dpf=sp.lambdify((r,theta),dp,'numpy'))


def compute(m, case, polar=False):
    a,b,c,d = map(float,case)
    if not np.all(np.isfinite([a,b,c,d])): raise ValueError('Entradas no finitas.')
    if polar:
        if a<=0: raise ValueError('El radio debe ser positivo.')
        # Se usa la rama principal de atan2; theta en [-pi, pi].
        if not -np.pi <= b <= np.pi: raise ValueError('Use theta en [-pi, pi].')
        R,T=a,b
        X,Y=R*np.cos(T),R*np.sin(T)
    else:
        X,Y=a,b
        R,T=np.hypot(X,Y),np.arctan2(Y,X)
    if R<=1e-12: raise ValueError('Evite el origen polar (r > 1e-12).')
    C,S=np.cos(T),np.sin(T)
    J=np.array([[C,-R*S],[S,R*C]]) # d(x,y)/d(r,theta)
    K=np.array([[C,S],[-S/R,C/R]]) # inversa
    G=np.diag([1.,R**2]); Gi=np.diag([1.,R**-2])
    if polar:
        vp=np.array([c,d]); vc=J@vp
    else:
        vc=np.array([c,d]); vp=K@vc
    # Diferenciación independiente en los dos sistemas.
    dc=np.asarray(m['dcf'](X,Y),dtype=float).reshape(2)
    dp=np.asarray(m['dpf'](R,T),dtype=float).reshape(2)
    gc=dc.copy(); gp=Gi@dp
    fc=float(m['fc'](X,Y)); fp=float(m['fp'](R,T))
    D1=float(dc@vc); D2=float(dp@vp)
    n1=float(np.linalg.norm(gc)); n2=float(np.sqrt(gp@G@gp))
    values=np.r_[dc,dp,gc,gp,fc,fp,D1,D2,n1,n2]
    if not np.all(np.isfinite(values)): raise ValueError('Función o derivadas no finitas en el punto.')
    return dict(X=X,Y=Y,R=R,T=T,J=J,K=K,G=G,Gi=Gi,vc=vc,vp=vp,dc=dc,dp=dp,gc=gc,gp=gp,
                fc=fc,fp=fp,D1=D1,D2=D2,n1=n1,n2=n2,ED=abs(D1-D2),
                EG=float(np.linalg.norm(gc-J@gp)),EN=abs(n1-n2),EF=abs(fc-fp),
                Edf=float(np.linalg.norm(dc-K.T@dp)),Ev=float(np.linalg.norm(vc-J@vp)))


def table(results):
    rows=[]
    for i,z in enumerate(results,1):
        row={'Prueba':i,'x':z['X'],'y':z['Y'],'r':z['R'],'theta (rad)':z['T'],
             'f cart':z['fc'],'f polar':z['fp']}
        for key,labels in [('dc',('df_x','df_y')),('dp',('df_r','df_theta')),
                           ('gc',('grad^x','grad^y')),('gp',('grad^r','grad^theta')),
                           ('vc',('v^x','v^y')),('vp',('v^r','v^theta'))]:
            row.update(zip(labels,z[key]))
        row.update({'D cart':z['D1'],'D polar':z['D2'],'norma grad cart':z['n1'],
                    'norma grad polar':z['n2'],'E_D':z['ED'],'E_grad':z['EG'],
                    'E_norma':z['EN'],'E_f':z['EF'],'E_df':z['Edf'],'E_v':z['Ev']})
        rows.append(row)
    return pd.DataFrame(rows)


def plot(m,results):
    L=max(2.5,max(max(abs(z['X']),abs(z['Y'])) for z in results)*1.5)
    a=np.linspace(-L,L,181); X,Y=np.meshgrid(a,a)
    with np.errstate(all='ignore'):
        Z=np.asarray(m['fc'](X,Y))
        if np.iscomplexobj(Z): raise ValueError('La gráfica requiere una función real.')
        Z=np.ma.masked_invalid(np.broadcast_to(Z,X.shape))
    if Z.count()==0: raise ValueError('No hay valores finitos en el dominio gráfico.')
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
    cf=ax.contourf(X,Y,Z,levels=24,cmap='viridis')
    fig.colorbar(cf,ax=ax,label='f(x,y)')
    ax.contour(X,Y,Z,levels=12,colors='white',linewidths=.45,alpha=.45)
    ax.axhline(0,color='white',alpha=.4); ax.axvline(0,color='white',alpha=.4)
    for i,z in enumerate(results,1):
        p=np.array([z['X'],z['Y']]); ax.plot([0,p[0]],[0,p[1]],'w--',lw=.8,alpha=.8)
        ax.scatter(*p,c='white',edgecolor='black',s=55,zorder=5)
        ax.annotate(f'P{i}',p,xytext=(6,8),textcoords='offset points',color='white',weight='bold')
        for vec,col,label in [(z['gc'],'#ffb454','Gradiente'),(z['vc'],'#7dd3fc','Dirección v')]:
            n=np.linalg.norm(vec)
            if n>0:
                u=vec/n*(L*.23)
                ax.quiver(*p,*u,angles='xy',scale_units='xy',scale=1,color=col,width=.008,
                          label=label if i==1 else None,zorder=6)
    ax.set(xlim=(-L,L),ylim=(-L,L),xlabel='x',ylabel='y',title='Potencial escalar y vectores en los puntos de prueba',aspect='equal')
    ax.legend(loc='upper left',title='Flechas normalizadas solo para visualizar')
    return fig


def app():
    import streamlit as st
    st.set_page_config(page_title='PIA | Cálculo tensorial',layout='wide')
    st.title('Gradiente y diferencial en coordenadas cartesianas y polares')
    st.caption('R² euclidiano · Base coordenada polar (no ortonormal) · Ángulos en radianes')
    fmode=st.radio('Sistema para definir el potencial',['Cartesiano','Polar'],horizontal=True)
    pfunc=fmode=='Polar'
    expr=st.text_input('Función potencial (sintaxis Python/SymPy)',
                       'r**2*(cos(theta)**2+2*sin(theta)**2+cos(theta)*sin(theta))' if pfunc else DEFAULT,key=f'f_{fmode}')
    mode=st.radio('Sistema de entrada de puntos y vectores',['Cartesiano','Polar'],horizontal=True)
    polar=mode=='Polar'
    st.caption('v contiene componentes contravariantes de la base coordenada. No se normaliza para la contracción.')
    names=['r','theta','v^r','v^theta'] if polar else ['x','y','v^x','v^y']
    defaults=[(np.hypot(a,b),np.arctan2(b,a),(a*c+b*d)/np.hypot(a,b),(a*d-b*c)/(a*a+b*b)) for a,b,c,d in CASES] if polar else CASES
    edited=st.data_editor(pd.DataFrame(defaults,columns=names),num_rows='dynamic',key=f'cases_{mode}',use_container_width=True)
    try:
        m=model(expr,pfunc)
        if len(edited)<3: raise ValueError('Agregue al menos tres pruebas.')
        results=[compute(m,row,polar) for row in edited.itertuples(index=False,name=None)]
        st.subheader('Desarrollo simbólico')
        for label,key in [('f(x,y)','f'),('F(r,\\theta)','F')]:
            st.latex(label+'='+sp.latex(m[key]))
        for key,coords in [('dc',['x','y']),('dp',['r','\\theta'])]:
            st.latex('df='+ '+'.join('('+sp.latex(q)+')\\,d'+c for q,c in zip(m[key],coords)))
        st.latex(r'g_{cart}=g^{cart}=I,\qquad g_{polar}=\mathrm{diag}(1,r^2),\quad g^{polar}=\mathrm{diag}(1,r^{-2})')
        st.latex(r'(\nabla f)_{polar}^{i}=(\partial_r F,\,r^{-2}\partial_\theta F)')
        st.subheader('Resultados comparativos')
        df=table(results)
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button('Descargar tabla CSV',df.to_csv(index=False).encode('utf-8-sig'),'resultados_tensoriales.csv','text/csv')
        i=st.selectbox('Ver matrices y transformaciones de la prueba',range(1,len(results)+1))
        z=results[i-1]; cols=st.columns(2)
        for k,title,box in [('J','Jacobiana J: polar → cartesiano',cols[0]),('K','Jacobiana inversa K',cols[1]),('G','Métrica polar g',cols[0]),('Gi','Métrica inversa polar',cols[1])]:
            box.markdown('**'+title+'**'); box.dataframe(pd.DataFrame(z[k]),hide_index=True)
        st.write('Covector polar por transformación Jᵀ df_cart:',z['J'].T@z['dc'])
        st.write('Covector cartesiano por transformación Kᵀ df_polar:',z['K'].T@z['dp'])
        st.write('Gradiente polar devuelto a cartesianas J grad_polar:',z['J']@z['gp'])
        st.write('Vector cartesiano reconstruido J v_polar:',z['J']@z['vp'])
        st.pyplot(plot(m,results)); plt.close('all')
        st.info('E_grad compara vectores expresados en la misma base. E_norma compara sus normas métricas. '
                'Los ángulos usan atan2; para potenciales polares no periódicos, evite la discontinuidad de rama en el eje x negativo.')
    except Exception as e:
        st.error(str(e))


def demo():
    m=model(DEFAULT); results=[compute(m,c) for c in CASES]
    for z in results:
        for key in ('ED','EG','EN','EF','Edf','Ev'): assert z[key]<1e-10,(key,z[key])
        assert np.allclose(z['J'].T@z['J'],z['G'])
        reverse=compute(m,(z['R'],z['T'],*z['vp']),True)
        assert np.allclose(reverse['vc'],z['vc']) and np.isclose(reverse['D1'],z['D1'])
    mp=model('r**2*(cos(theta)**2+2*sin(theta)**2+cos(theta)*sin(theta))',True)
    for case,z in zip(CASES,results):
        zz=compute(mp,case)
        assert np.isclose(zz['fc'],z['fc']) and np.isclose(zz['D2'],z['D2'])
    root=Path(__file__).resolve().parent
    df=table(results); df.to_csv(root/'resultados_tensoriales.csv',index=False,encoding='utf-8-sig')
    fig=plot(m,results); fig.savefig(root/'grafica_tensorial.png',dpi=180); plt.close(fig)
    print(df.to_string(index=False))

if __name__=='__main__':
    if '--demo' in sys.argv: demo()
    else: app()
