"""Original architecture test fixture; not a deployment entry point."""

def to_risk_space(df):
    Z = df.copy()
    if "PredScore" in Z: Z["PredScore"] = -pd.to_numeric(Z["PredScore"], errors="coerce")
    if "Margin"    in Z: Z["Margin"]    = -pd.to_numeric(Z["Margin"],    errors="coerce")
    for c in META_FEATS_ALL:
        if c in Z:
            Z[c] = pd.to_numeric(Z[c], errors="coerce")
    return Z

def fit_meta_train_on(loo_train, loso_train, feats, fpr_target=0.20, seed=7):
    L = loo_train.copy();  L["dataset"]="LOO";  L["error"] = 1 - L["correct"].astype(int)
    S = loso_train.copy(); S["dataset"]="LOSO"; S["error"] = 1 - S["correct"].astype(int)
    D = pd.concat([L,S], ignore_index=True)
    D = to_risk_space(D)

    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs", random_state=seed)
    )
    clf = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    clf.fit(D[feats].fillna(0.0).values, D["error"].values)
    D["p_error"] = clf.predict_proba(D[feats].fillna(0.0).values)[:,1]

    Lcorr = D[(D["dataset"]=="LOO") & (D["error"]==0)]
    thr = float(np.nanpercentile(Lcorr["p_error"], 100*(1.0 - fpr_target))) if len(Lcorr)>=5 else 0.5
    return clf, thr
