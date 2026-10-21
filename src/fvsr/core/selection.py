"""R8 gamma decision extracted without any service/data side effects."""
GAMMAS=(0.,.25,.5,.75,1.)
def select_gamma(summary,native,old_gamma):
    old=summary[old_gamma];selected=old_gamma;bestJ=.5*(old['R100']+old['R10']);checks=[]
    for gamma in GAMMAS:
        m=summary[gamma];J=.5*(m['R100']+m['R10'])
        safe=all(m[k]-x[k]>=margin for x in (old,native) for k,margin in (('R100',-.001),('R10',-.0025)))
        checks.append(dict(gamma=gamma,J=J,protected=safe,metrics=m))
        if safe and J>bestJ:selected=gamma;bestJ=J
    return dict(selected_gamma=selected,changed=selected!=old_gamma,selection_checks=checks,
                rule='R8 strict J improvement; protection vs old deployment and Native; exact ties retain old gamma')

