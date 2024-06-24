classdef ControlFunctions
    methods(Static)
        
        function K = lqr(A, B, Q, R)
            % Método para calcular o regulador quadrático linear (LQR)
            [K, ~] = lqr(A, B, Q, R);
        end

        function plotResponse(sys)
            % Método para plotar a resposta do sistema
            [y, t] = step(sys);
            figure;
            plot(t, y);
            title('Resposta ao Degrau do Sistema');
            xlabel('Tempo (s)');
            ylabel('Resposta');
            grid on;
        end
        
        function sys = createSS(A, B, C, D)
            % Método para criar um sistema de espaço de estados
            sys = ss(A, B, C, D);
        end

        function [t, response] = openLoopResponse(amplitude, timeEnd)
            % Método para simular uma resposta de laço aberto hipotética
            t = linspace(0, timeEnd, 100);
            response = amplitude * (1 - exp(-t));
        end
        
        function plotPolesAndZeros(sys)
            % Método para plotar polos e zeros do sistema
            figure;
            pzmap(sys);
            title('Mapa de Polos e Zeros');
            grid on;
        end

        function result = exemploMetodo(param1, param2)
            % Método de exemplo que soma dois parâmetros
            result = param1 + param2;
        end
    end
end
