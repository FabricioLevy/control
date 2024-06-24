function main()
    % Definindo constantes e parâmetros
    g = 9.8; % aceleração da gravidade em m/s^2
    M_vazio = 29000; % massa vazia em kg
    L = 2.9; % comprimento em metros
    C = 17; % largura em metros
    H = 2; % altura em metros
    rho_fluido = 715; % densidade do fluido em kg/m^3
    hf = 1.2; % altura do fluido em metros
    mf = rho_fluido * L * C * hf; % massa do fluido

    m0 = 0.25 * mf;
    m1 = 0.25 * mf;
    m2 = 0.25 * mf;
    m3 = 0.25 * mf;

    k1 = 15000;
    k2 = 15000;
    
    h0 = 0; % altura de m0 em metros
    h1 = 0.6; % altura de m1 em metros
    h2 = 1.2; % altura de m2 em metros
    h3 = 1.8; % altura de m3 em metros

    % Cálculo da massa total e centro de massa
    Mt = M_vazio + m0 + m1 + m2 + m3;
    Hcm = (M_vazio + m0 * (1 + h0) + m1 * (1 + h1) + m2 * (1 + h2) + m3 * (1 + h3)) / Mt;
    
    % Cálculo da aceleração centrípeta
    velocidade_kmh = 80; % km/h
    velocidade_ms = velocidade_kmh / 3.6; % m/s
    raio = 300; % m
    aceleracao_centripeta = (velocidade_ms ^ 2) / raio;

    % Cálculo da força centrípeta
    Fc = Mt * aceleracao_centripeta;
    My = 2 * Hcm * Fc; % Momento resultante

    % Mostrar resultados
    disp(['Hcm: ', num2str(Hcm)]);
    disp(['Fc: ', num2str(Fc)]);
    disp(['My: ', num2str(My)]);
end
