import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders moderation dashboard heading', () => {
  render(<App />);
  const heading = screen.getByText(/SoulMath AI Moderation System/i);
  expect(heading).toBeInTheDocument();
});
